import os
from pathlib import Path

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


# ============================================================
# TRUTHLENS - BASELINE MODEL TRAINING
# EfficientNetB0 Transfer Learning
# ============================================================

# -------------------- PATHS ---------------------------------

PROJECT_DIR = Path(r"D:\TruthLens")

DATASET_DIR = PROJECT_DIR / "Final_Dataset"

TRAIN_DIR = DATASET_DIR / "Train"
VAL_DIR = DATASET_DIR / "Validation"
TEST_DIR = DATASET_DIR / "Test"

MODEL_DIR = PROJECT_DIR / "Models"

RESULTS_DIR = PROJECT_DIR / "Results"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# -------------------- SETTINGS ------------------------------

IMAGE_SIZE = (224, 224)

BATCH_SIZE = 16

EPOCHS = 5

LEARNING_RATE = 0.001

SEED = 42

NUM_CLASSES = 4


# ============================================================
# SYSTEM INFORMATION
# ============================================================

print("=" * 70)
print("TRUTHLENS - EFFICIENTNETB0 BASELINE TRAINING")
print("=" * 70)

print("\nTensorFlow version:")
print(tf.__version__)

gpu_devices = tf.config.list_physical_devices("GPU")

print("\nGPU devices detected:")

if gpu_devices:

    for gpu in gpu_devices:
        print(f"  {gpu}")

else:

    print("  No GPU detected.")
    print("  TensorFlow will use CPU.")


# ============================================================
# CHECK DATASET
# ============================================================

print("\n" + "=" * 70)
print("CHECKING DATASET")
print("=" * 70)

required_directories = [
    TRAIN_DIR,
    VAL_DIR,
    TEST_DIR
]

for directory in required_directories:

    if not directory.exists():

        raise FileNotFoundError(
            f"Dataset directory not found:\n{directory}"
        )

    print(f"Found: {directory}")


# ============================================================
# LOAD DATASETS
# ============================================================

print("\n" + "=" * 70)
print("LOADING DATASETS")
print("=" * 70)


train_dataset = keras.utils.image_dataset_from_directory(

    TRAIN_DIR,

    labels="inferred",

    label_mode="int",

    image_size=IMAGE_SIZE,

    batch_size=BATCH_SIZE,

    shuffle=True,

    seed=SEED
)


validation_dataset = keras.utils.image_dataset_from_directory(

    VAL_DIR,

    labels="inferred",

    label_mode="int",

    image_size=IMAGE_SIZE,

    batch_size=BATCH_SIZE,

    shuffle=False
)


test_dataset = keras.utils.image_dataset_from_directory(

    TEST_DIR,

    labels="inferred",

    label_mode="int",

    image_size=IMAGE_SIZE,

    batch_size=BATCH_SIZE,

    shuffle=False
)


# ============================================================
# CLASS NAMES
# ============================================================

class_names = train_dataset.class_names

print("\nClass mapping:")

for index, class_name in enumerate(class_names):

    print(
        f"  {index} -> {class_name}"
    )


if len(class_names) != NUM_CLASSES:

    raise ValueError(
        f"Expected {NUM_CLASSES} classes, "
        f"but found {len(class_names)}."
    )


# ============================================================
# PERFORMANCE OPTIMIZATION
# ============================================================

AUTOTUNE = tf.data.AUTOTUNE


train_dataset = train_dataset.prefetch(
    buffer_size=AUTOTUNE
)

validation_dataset = validation_dataset.prefetch(
    buffer_size=AUTOTUNE
)

test_dataset = test_dataset.prefetch(
    buffer_size=AUTOTUNE
)


# ============================================================
# DATA AUGMENTATION
# ============================================================

data_augmentation = keras.Sequential(

    [

        layers.RandomFlip(
            "horizontal"
        ),

        layers.RandomRotation(
            0.05
        ),

        layers.RandomZoom(
            0.10
        ),

        layers.RandomContrast(
            0.10
        )

    ],

    name="data_augmentation"
)


# ============================================================
# LOAD EFFICIENTNETB0
# ============================================================

print("\n" + "=" * 70)
print("BUILDING MODEL")
print("=" * 70)


base_model = keras.applications.EfficientNetB0(

    weights="imagenet",

    include_top=False,

    input_shape=(
        IMAGE_SIZE[0],
        IMAGE_SIZE[1],
        3
    )
)


# Freeze EfficientNet initially
base_model.trainable = False


# ============================================================
# BUILD CLASSIFIER
# ============================================================

inputs = keras.Input(
    shape=(
        IMAGE_SIZE[0],
        IMAGE_SIZE[1],
        3
    )
)


# Data augmentation only runs during training
x = data_augmentation(inputs)


# IMPORTANT:
# EfficientNetB0 in modern tf.keras already includes its
# required input rescaling internally.
#
# Therefore we do NOT manually divide pixels by 255 here.


x = base_model(
    x,
    training=False
)


x = layers.GlobalAveragePooling2D()(x)


x = layers.Dropout(
    0.30
)(x)


outputs = layers.Dense(

    NUM_CLASSES,

    activation="softmax"

)(x)


model = keras.Model(

    inputs,

    outputs,

    name="TruthLens_EfficientNetB0"
)


# ============================================================
# COMPILE MODEL
# ============================================================

model.compile(

    optimizer=keras.optimizers.Adam(
        learning_rate=LEARNING_RATE
    ),

    loss=keras.losses.SparseCategoricalCrossentropy(),

    metrics=[
        "accuracy"
    ]
)


# ============================================================
# MODEL SUMMARY
# ============================================================

print("\nModel architecture:\n")

model.summary()


# ============================================================
# CALLBACKS
# ============================================================

BEST_MODEL_PATH = (
    MODEL_DIR
    / "truthlens_efficientnetb0_best.keras"
)


callbacks = [

    # Save only the best validation model
    keras.callbacks.ModelCheckpoint(

        filepath=str(BEST_MODEL_PATH),

        monitor="val_accuracy",

        mode="max",

        save_best_only=True,

        verbose=1
    ),


    # Stop if validation loss stops improving
    keras.callbacks.EarlyStopping(

        monitor="val_loss",

        patience=4,

        restore_best_weights=True,

        verbose=1
    ),


    # Reduce learning rate when improvement stalls
    keras.callbacks.ReduceLROnPlateau(

        monitor="val_loss",

        factor=0.2,

        patience=2,

        min_lr=1e-7,

        verbose=1
    ),


    # Save epoch information to CSV
    keras.callbacks.CSVLogger(

        str(
            RESULTS_DIR
            / "baseline_training_log.csv"
        )
    )
]


# ============================================================
# TRAIN MODEL
# ============================================================

print("\n" + "=" * 70)
print("STARTING BASELINE TRAINING")
print("=" * 70)

print(f"\nImage size    : {IMAGE_SIZE}")
print(f"Batch size    : {BATCH_SIZE}")
print(f"Max epochs    : {EPOCHS}")
print(f"Learning rate : {LEARNING_RATE}")
print(f"Classes       : {NUM_CLASSES}")

print(
    "\nBest model will be saved to:"
)

print(BEST_MODEL_PATH)


history = model.fit(

    train_dataset,

    validation_data=validation_dataset,

    epochs=EPOCHS,

    callbacks=callbacks
)


# ============================================================
# SAVE FINAL MODEL
# ============================================================

FINAL_MODEL_PATH = (
    MODEL_DIR
    / "truthlens_efficientnetb0_final.keras"
)


model.save(
    FINAL_MODEL_PATH
)


print(
    "\nFinal model saved:"
)

print(FINAL_MODEL_PATH)


# ============================================================
# TEST DATASET EVALUATION
# ============================================================

print("\n" + "=" * 70)
print("BASELINE TEST EVALUATION")
print("=" * 70)


test_loss, test_accuracy = model.evaluate(

    test_dataset,

    verbose=1
)


print(
    f"\nTest Loss     : "
    f"{test_loss:.4f}"
)

print(
    f"Test Accuracy : "
    f"{test_accuracy:.4f}"
)

print(
    f"Test Accuracy : "
    f"{test_accuracy * 100:.2f}%"
)


# ============================================================
# SAVE BASIC TEST RESULTS
# ============================================================

result_file = (
    RESULTS_DIR
    / "baseline_test_results.txt"
)


with open(
    result_file,
    "w",
    encoding="utf-8"
) as file:

    file.write(
        "TruthLens EfficientNetB0 Baseline\n"
    )

    file.write(
        "================================\n"
    )

    file.write(
        f"Classes: {class_names}\n"
    )

    file.write(
        f"Test Loss: {test_loss:.6f}\n"
    )

    file.write(
        f"Test Accuracy: "
        f"{test_accuracy:.6f}\n"
    )

    file.write(
        f"Test Accuracy Percentage: "
        f"{test_accuracy * 100:.2f}%\n"
    )


print(
    "\nResults saved:"
)

print(result_file)


print("\n" + "=" * 70)
print("BASELINE TRAINING COMPLETED")
print("=" * 70)