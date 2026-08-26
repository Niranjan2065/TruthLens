"""
TruthLens - Screen-Capture Domain Adaptation Fine-Tuning
=========================================================

Purpose
-------
Improve real-world generalization of the existing TruthLens mobile-adapted
EfficientNetB0 model without using the final RealWorld_Test set.

Baseline:
    Internal accuracy      : 85.54%
    Real-world accuracy    : 66.75%
    Manipulated recall     : 6.00%

Training:
    data/train_targeted
    data/val_holdout

Final test set:
    RealWorld_Test
    NEVER used by this script.

Important fixes in this version
--------------------------------
1. Augmentation is applied to ONE image at a time, not to an entire batch.
2. The batch dimension is restored after augmentation.
3. Images are kept in 0-255 format while PIL/TensorFlow augmentation runs.
4. Images are converted to 0-1 immediately before being passed to the model.
5. Validation data is NEVER augmented.
6. RealWorld_Test path is blocked by a safety check.
7. --real_weight is supported.
8. The best checkpoint is reloaded before validation evaluation.

Example:
    python screen_capture_finetune.py --base_model "D:\\TruthLens\\Models\\truthlens_mobile_adapted.keras" --train_dir "D:\\TruthLens\\data\\train_targeted" --val_dir "D:\\TruthLens\\data\\val_holdout" --output "D:\\TruthLens\\Models\\truthlens_v2_domain_adapted.keras" --manipulated_weight 3.0 --real_weight 1.2 --epochs 20 --lr 0.00003
"""

import argparse
import io
import os
import random

import numpy as np
import tensorflow as tf
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix


# ============================================================
# CONFIGURATION
# ============================================================

IMG_SIZE = (224, 224)
BATCH_SIZE = 16

CLASS_NAMES = [
    "AI_Generated",
    "Deepfake",
    "Manipulated",
    "Real",
]

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)


# ============================================================
# SCREEN-CAPTURE AUGMENTATION
# ============================================================

def jpeg_recompress(image, quality_range=(60, 90)):
    """
    Simulate JPEG recompression.

    Input:
        image: [224, 224, 3], float32, 0-255

    Output:
        image: [224, 224, 3], float32, 0-255
    """
    image = np.asarray(image, dtype=np.float32)
    image_uint8 = np.clip(image, 0.0, 255.0).astype(np.uint8)

    pil_image = Image.fromarray(image_uint8, mode="RGB")

    quality = random.randint(*quality_range)

    buffer = io.BytesIO()
    pil_image.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)

    with Image.open(buffer) as decoded:
        result = decoded.convert("RGB")
        result = np.asarray(result, dtype=np.float32)

    return result


def moire_aliasing(image, scale_range=(0.55, 0.85)):
    """
    Simulate downscale/upscale artifacts.

    Input/output remain 224x224x3 and 0-255.
    """
    image = tf.convert_to_tensor(image, dtype=tf.float32)

    shape = tf.shape(image)
    h = int(shape[0].numpy())
    w = int(shape[1].numpy())

    scale = random.uniform(*scale_range)

    new_h = max(1, int(h * scale))
    new_w = max(1, int(w * scale))

    small = tf.image.resize(
        image,
        [new_h, new_w],
        method="area",
    )

    restored = tf.image.resize(
        small,
        [h, w],
        method="bilinear",
    )

    return restored.numpy().astype(np.float32)


def color_space_shift(image, max_delta=0.06):
    """
    Apply a mild hue/saturation shift.

    Input/output remain 0-255.
    """
    image = tf.convert_to_tensor(image, dtype=tf.float32)
    image = tf.clip_by_value(image / 255.0, 0.0, 1.0)

    image = tf.image.random_hue(
        image,
        max_delta=max_delta,
    )

    image = tf.image.random_saturation(
        image,
        lower=0.90,
        upper=1.10,
    )

    image = tf.clip_by_value(image, 0.0, 1.0)

    return (image.numpy() * 255.0).astype(np.float32)


def screen_capture_augment(image, probability=0.7):
    """
    Apply a random subset of screen-capture effects to ONE image.

    This function intentionally accepts one image only:
        [224, 224, 3]

    It must NOT receive:
        [batch, 224, 224, 3]
    """
    image = np.asarray(image, dtype=np.float32)

    if image.shape != (224, 224, 3):
        raise ValueError(
            f"Augmentation received unexpected shape: {image.shape}. "
            f"Expected (224, 224, 3)."
        )

    if random.random() < probability:
        image = moire_aliasing(image)

    if random.random() < probability:
        image = color_space_shift(image)

    if random.random() < probability:
        image = jpeg_recompress(image)

    return np.clip(image, 0.0, 255.0).astype(np.float32)


def augmentation_wrapper(image, label):
    """
    TensorFlow wrapper for ONE image.

    Input:
        image = [224,224,3], 0-255

    Output:
        image = [224,224,3], 0-255
    """
    augmented = tf.py_function(
        func=screen_capture_augment,
        inp=[image],
        Tout=tf.float32,
    )

    augmented.set_shape((224, 224, 3))

    return augmented, label


# ============================================================
# DATASET
# ============================================================

def build_dataset(directory, training=False):
    """
    Build a dataset with a guaranteed shape:

        images: [batch, 224, 224, 3]
        labels: [batch, 4]

    Training:
        - shuffle
        - screen-capture augmentation
        - normalize to 0-1

    Validation:
        - no augmentation
        - normalize to 0-1
    """

    print()
    print("=" * 70)
    print("LOADING DATASET")
    print("=" * 70)
    print("Directory:", directory)

    dataset = tf.keras.utils.image_dataset_from_directory(
        directory,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="categorical",
        class_names=CLASS_NAMES,
        shuffle=training,
        seed=SEED,
    )

    print("Dataset loaded.")

    if training:
        print("Screen-capture augmentation: ENABLED")

        # image_dataset_from_directory produces:
        # [batch, 224, 224, 3]
        #
        # Unbatch so augmentation receives exactly:
        # [224, 224, 3]
        dataset = dataset.unbatch()

        dataset = dataset.map(
            augmentation_wrapper,
            num_parallel_calls=tf.data.AUTOTUNE,
        )

        # Restore the batch dimension.
        dataset = dataset.batch(BATCH_SIZE)

        # Normalize after augmentation.
        dataset = dataset.map(
            lambda x, y: (
                tf.cast(x, tf.float32) / 255.0,
                y,
            ),
            num_parallel_calls=tf.data.AUTOTUNE,
        )

    else:
        print("Screen-capture augmentation: DISABLED")

        # Validation images are not modified.
        dataset = dataset.map(
            lambda x, y: (
                tf.cast(x, tf.float32) / 255.0,
                y,
            ),
            num_parallel_calls=tf.data.AUTOTUNE,
        )

    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset


# ============================================================
# MODEL PREPARATION
# ============================================================

def find_efficientnet_backbone(model):
    """
    Find an EfficientNet backbone in the loaded model.
    """
    for layer in model.layers:
        if "efficientnet" in layer.name.lower():
            return layer

    return None


def prepare_model(model_path):
    """
    Load the existing mobile-adapted model.

    Freeze the model first, then unfreeze only the last 30
    EfficientNet backbone layers (excluding BatchNorm), while
    keeping the classifier head trainable.
    """

    print()
    print("=" * 70)
    print("LOADING BASE MODEL")
    print("=" * 70)
    print("Model:", model_path)

    if not os.path.isfile(model_path):
        raise FileNotFoundError(
            f"Base model was not found:\n{model_path}"
        )

    model = tf.keras.models.load_model(model_path)

    print()
    print("Original input :", model.input_shape)
    print("Original output:", model.output_shape)

    if tuple(model.input_shape[1:]) != (224, 224, 3):
        raise ValueError(
            f"Expected model input (224,224,3), "
            f"but found {model.input_shape}"
        )

    if model.output_shape[-1] != 4:
        raise ValueError(
            f"Expected 4 output classes, "
            f"but found {model.output_shape}"
        )

    # Freeze everything first.
    for layer in model.layers:
        layer.trainable = False

    backbone = find_efficientnet_backbone(model)

    if backbone is None:
        raise RuntimeError(
            "EfficientNet backbone could not be found in the model."
        )

    print()
    print("Backbone found:", backbone.name)

    # Freeze the whole backbone first.
    backbone.trainable = False

    for layer in backbone.layers:
        layer.trainable = False

    # Unfreeze the last 30 layers.
    trainable_backbone_count = 0

    for layer in backbone.layers[-30:]:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False
        else:
            layer.trainable = True
            trainable_backbone_count += 1

    # Keep the top classifier/head trainable.
    # We do not rely on exact layer names because custom model
    # heads can have names such as dense_1, dropout_1, etc.
    for layer in model.layers:
        if layer is backbone:
            continue

        if isinstance(
            layer,
            (
                tf.keras.layers.Dense,
                tf.keras.layers.Dropout,
                tf.keras.layers.GlobalAveragePooling2D,
                tf.keras.layers.GlobalMaxPooling2D,
                tf.keras.layers.Flatten,
            ),
        ):
            layer.trainable = True

    print()
    print(
        "Trainable backbone layers:",
        trainable_backbone_count,
    )

    total_trainable = sum(
        int(np.prod(v.shape))
        for v in model.trainable_variables
    )

    print(
        "Trainable parameters:",
        f"{total_trainable:,}",
    )

    return model


# ============================================================
# COMPILE
# ============================================================

def compile_model(model, learning_rate):
    print()
    print("=" * 70)
    print("COMPILING MODEL")
    print("=" * 70)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=learning_rate
        ),
        loss="categorical_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.F1Score(
                average="macro",
                name="macro_f1",
            ),
        ],
    )

    print("Learning rate:", learning_rate)
    print("Optimizer: Adam")
    print("Loss: categorical_crossentropy")
    print("Metric: accuracy + macro F1")


# ============================================================
# TRAINING
# ============================================================

def train_model(
    model,
    train_dir,
    val_dir,
    output_path,
    manipulated_weight,
    real_weight,
    epochs,
):
    train_dataset = build_dataset(
        train_dir,
        training=True,
    )

    val_dataset = build_dataset(
        val_dir,
        training=False,
    )

    # Verify dataset shapes before starting a potentially long run.
    train_spec = train_dataset.element_spec
    val_spec = val_dataset.element_spec

    print()
    print("=" * 70)
    print("DATASET SHAPE CHECK")
    print("=" * 70)

    print("Train images:", train_spec[0].shape)
    print("Train labels:", train_spec[1].shape)
    print("Val images  :", val_spec[0].shape)
    print("Val labels  :", val_spec[1].shape)

    expected_image_shape = (None, 224, 224, 3)

    if train_spec[0].shape.rank != 4:
        raise RuntimeError(
            f"Training dataset has wrong rank: "
            f"{train_spec[0].shape}. "
            f"Expected [batch,224,224,3]."
        )

    if val_spec[0].shape.rank != 4:
        raise RuntimeError(
            f"Validation dataset has wrong rank: "
            f"{val_spec[0].shape}. "
            f"Expected [batch,224,224,3]."
        )

    # Class order:
    # 0 = AI_Generated
    # 1 = Deepfake
    # 2 = Manipulated
    # 3 = Real

    class_weights = {
        0: 1.0,
        1: 1.0,
        2: manipulated_weight,
        3: real_weight,
    }

    print()
    print("=" * 70)
    print("CLASS WEIGHTS")
    print("=" * 70)
    print("AI_Generated :", class_weights[0])
    print("Deepfake     :", class_weights[1])
    print("Manipulated  :", class_weights[2])
    print("Real         :", class_weights[3])

    output_dir = os.path.dirname(output_path)

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_macro_f1",
            mode="max",
            patience=5,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_macro_f1",
            mode="max",
            factor=0.5,
            patience=2,
            min_lr=1e-7,
            verbose=1,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=output_path,
            monitor="val_macro_f1",
            mode="max",
            save_best_only=True,
            verbose=1,
        ),
    ]

    print()
    print("=" * 70)
    print("STARTING FINE-TUNING")
    print("=" * 70)
    print("Epochs:", epochs)
    print("Batch size:", BATCH_SIZE)

    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=epochs,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1,
    )

    return history


# ============================================================
# VALIDATION EVALUATION
# ============================================================

def evaluate_model(model, val_dir):
    print()
    print("=" * 70)
    print("VALIDATION EVALUATION")
    print("=" * 70)

    dataset = build_dataset(
        val_dir,
        training=False,
    )

    y_true = []
    y_pred = []

    for images, labels in dataset:
        predictions = model.predict(
            images,
            verbose=0,
        )

        y_true.extend(
            np.argmax(
                labels.numpy(),
                axis=1,
            )
        )

        y_pred.extend(
            np.argmax(
                predictions,
                axis=1,
            )
        )

    print()
    print("CONFUSION MATRIX")
    print("-" * 70)
    print("Rows = True class")
    print("Columns = Predicted class")
    print()
    print("Class order:")
    print(CLASS_NAMES)

    matrix = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1, 2, 3],
    )

    print()
    print(matrix)

    print()
    print("CLASSIFICATION REPORT")
    print("-" * 70)

    report = classification_report(
        y_true,
        y_pred,
        labels=[0, 1, 2, 3],
        target_names=CLASS_NAMES,
        digits=4,
        zero_division=0,
    )

    print(report)

    print()
    print("=" * 70)
    print("PER-CLASS RECALL")
    print("=" * 70)

    for i, class_name in enumerate(CLASS_NAMES):
        total = int(matrix[i].sum())
        correct = int(matrix[i, i])

        recall = (
            correct / total
            if total > 0
            else 0.0
        )

        print(
            f"{class_name:<15}: "
            f"{correct}/{total} "
            f"({recall * 100:.2f}%)"
        )

    total_samples = int(matrix.sum())
    total_correct = int(np.trace(matrix))

    accuracy = (
        total_correct / total_samples
        if total_samples > 0
        else 0.0
    )

    print()
    print(
        f"Validation Accuracy: "
        f"{accuracy * 100:.2f}%"
    )

    return matrix


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="TruthLens screen-capture domain adaptation"
    )

    parser.add_argument(
        "--base_model",
        required=True,
        help="Path to truthlens_mobile_adapted.keras",
    )

    parser.add_argument(
        "--train_dir",
        required=True,
        help="Path to data/train_targeted",
    )

    parser.add_argument(
        "--val_dir",
        required=True,
        help="Path to data/val_holdout",
    )

    parser.add_argument(
        "--output",
        default="Models/truthlens_v2_domain_adapted.keras",
        help="Output path for the best model",
    )

    parser.add_argument(
        "--manipulated_weight",
        type=float,
        default=3.0,
        help="Class weight for Manipulated",
    )

    parser.add_argument(
        "--real_weight",
        type=float,
        default=1.2,
        help="Class weight for Real",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
        help="Maximum number of epochs",
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=3e-5,
        help="Learning rate",
    )

    args = parser.parse_args()

    print("=" * 70)
    print("TRUTHLENS - SCREEN-CAPTURE DOMAIN ADAPTATION")
    print("=" * 70)

    print()
    print("Base model:")
    print(args.base_model)

    print()
    print("Training data:")
    print(args.train_dir)

    print()
    print("Validation data:")
    print(args.val_dir)

    print()
    print("Output:")
    print(args.output)

    print()
    print("IMPORTANT:")
    print("RealWorld_Test is NOT used during training.")

    # --------------------------------------------------------
    # Safety check: never allow RealWorld_Test as train/val.
    # --------------------------------------------------------
    forbidden_tokens = [
        "realworld_test",
        "realworldtest",
    ]

    for path in [
        args.train_dir,
        args.val_dir,
    ]:
        normalized = os.path.normpath(path).lower()

        for token in forbidden_tokens:
            if token in normalized:
                raise RuntimeError(
                    "\nSAFETY STOP\n"
                    "RealWorld_Test must NEVER be used as "
                    "training or validation data.\n"
                    f"Rejected path: {path}"
                )

    # --------------------------------------------------------
    # Prepare and compile model.
    # --------------------------------------------------------
    model = prepare_model(
        args.base_model
    )

    compile_model(
        model,
        args.lr
    )

    # --------------------------------------------------------
    # Train.
    # --------------------------------------------------------
    history = train_model(
        model=model,
        train_dir=args.train_dir,
        val_dir=args.val_dir,
        output_path=args.output,
        manipulated_weight=args.manipulated_weight,
        real_weight=args.real_weight,
        epochs=args.epochs,
    )

    # --------------------------------------------------------
    # Best checkpoint should have been written by
    # ModelCheckpoint.
    # --------------------------------------------------------
    print()
    print("=" * 70)
    print("BEST MODEL")
    print("=" * 70)
    print("Saved to:", args.output)

    if not os.path.isfile(args.output):
        raise RuntimeError(
            "Training finished, but the best model file was not found:\n"
            f"{args.output}"
        )

    print()
    print("Reloading best checkpoint...")

    best_model = tf.keras.models.load_model(
        args.output
    )

    # --------------------------------------------------------
    # Evaluate ONLY on val_holdout.
    # RealWorld_Test remains untouched.
    # --------------------------------------------------------
    evaluate_model(
        best_model,
        args.val_dir
    )

    print()
    print("=" * 70)
    print("FINE-TUNING COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
