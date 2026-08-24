import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from pathlib import Path

# ============================================================
# TRUTHLENS - MOBILE ADAPTED MODEL FINE-TUNING
# ============================================================

BASE_DIR = Path(r"D:\TruthLens")

DATASET_DIR = BASE_DIR / "Mobile_Adapted_Training"

BASE_MODEL_PATH = (
    BASE_DIR / "Models" /
    "truthlens_efficientnetb0_finetuned_final.keras"
)

OUTPUT_DIR = BASE_DIR / "Models"
OUTPUT_MODEL_PATH = (
    OUTPUT_DIR / "truthlens_mobile_adapted.keras"
)

IMG_SIZE = (224, 224)
BATCH_SIZE = 16
SEED = 42

CLASS_NAMES = [
    "AI_Generated",
    "Deepfake",
    "Manipulated",
    "Real"
]

print("=" * 70)
print("TRUTHLENS - MOBILE ADAPTED MODEL FINE-TUNING")
print("=" * 70)

print()
print("Dataset:")
print(DATASET_DIR)

print()
print("Base model:")
print(BASE_MODEL_PATH)

print()
print("Output model:")
print(OUTPUT_MODEL_PATH)

# ============================================================
# CHECK FILES
# ============================================================

if not DATASET_DIR.exists():
    raise FileNotFoundError(
        f"Dataset not found:\n{DATASET_DIR}"
    )

if not BASE_MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Base model not found:\n{BASE_MODEL_PATH}"
    )

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# CHECK DATASET
# ============================================================

print()
print("=" * 70)
print("CHECKING DATASET")
print("=" * 70)

for class_name in CLASS_NAMES:

    class_dir = DATASET_DIR / class_name

    if not class_dir.exists():
        raise FileNotFoundError(
            f"Missing class folder:\n{class_dir}"
        )

    count = len([
        p for p in class_dir.iterdir()
        if p.is_file()
        and p.suffix.lower() in
        [".jpg", ".jpeg", ".png", ".bmp", ".webp"]
    ])

    print(f"{class_name:<15}: {count}")

# ============================================================
# LOAD DATASET
# ============================================================

print()
print("=" * 70)
print("LOADING DATASET")
print("=" * 70)

train_ds = tf.keras.utils.image_dataset_from_directory(
    DATASET_DIR,
    labels="inferred",
    label_mode="categorical",
    class_names=CLASS_NAMES,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    validation_split=0.15,
    subset="training",
    seed=SEED,
    shuffle=True
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    DATASET_DIR,
    labels="inferred",
    label_mode="categorical",
    class_names=CLASS_NAMES,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    validation_split=0.15,
    subset="validation",
    seed=SEED,
    shuffle=False
)

print()
print("Class names:")
print(train_ds.class_names)

# ============================================================
# PERFORMANCE
# ============================================================

AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.prefetch(AUTOTUNE)
val_ds = val_ds.prefetch(AUTOTUNE)

# IMPORTANT:
# Do NOT divide images by 255 here.
#
# The existing TruthLens EfficientNet model expects the same
# input convention used during its original training.
#
# image_dataset_from_directory provides pixel values in 0-255.

# ============================================================
# LOAD EXISTING MODEL
# ============================================================

print()
print("=" * 70)
print("LOADING EXISTING TRUTHLENS MODEL")
print("=" * 70)

model = keras.models.load_model(BASE_MODEL_PATH)

print()
print("Model loaded successfully.")

print()
print("Input shape:")
print(model.input_shape)

print()
print("Output shape:")
print(model.output_shape)

# ============================================================
# SHOW LAYERS
# ============================================================

print()
print("=" * 70)
print("MODEL STRUCTURE")
print("=" * 70)

for i, layer in enumerate(model.layers):
    print(
        f"{i:3d} | "
        f"{layer.name:<35} | "
        f"{layer.__class__.__name__:<25} | "
        f"Trainable: {layer.trainable}"
    )

# ============================================================
# PHASE 1
# FREEZE EFFICIENTNET BACKBONE
# ============================================================

print()
print("=" * 70)
print("PHASE 1 - CLASSIFIER ADAPTATION")
print("=" * 70)

# Find EfficientNet backbone
efficientnet = None

for layer in model.layers:
    if layer.name == "efficientnetb0":
        efficientnet = layer
        break

if efficientnet is None:
    raise RuntimeError(
        "EfficientNetB0 backbone not found."
    )

efficientnet.trainable = False

# Make classifier trainable
for layer in model.layers:
    if layer.name in ["dense", "dropout"]:
        layer.trainable = True

print("EfficientNetB0: FROZEN")
print("Classifier: TRAINABLE")

# ============================================================
# COMPILE PHASE 1
# ============================================================

model.compile(
    optimizer=keras.optimizers.Adam(
        learning_rate=1e-4
    ),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# ============================================================
# CALLBACKS
# ============================================================

checkpoint_phase1 = keras.callbacks.ModelCheckpoint(
    str(OUTPUT_DIR / "truthlens_mobile_phase1.keras"),
    monitor="val_accuracy",
    save_best_only=True,
    verbose=1
)

early_stop_phase1 = keras.callbacks.EarlyStopping(
    monitor="val_accuracy",
    patience=3,
    restore_best_weights=True,
    verbose=1
)

reduce_lr_phase1 = keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=2,
    min_lr=1e-7,
    verbose=1
)

# ============================================================
# TRAIN PHASE 1
# ============================================================

print()
print("=" * 70)
print("STARTING PHASE 1 TRAINING")
print("=" * 70)

history1 = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=5,
    callbacks=[
        checkpoint_phase1,
        early_stop_phase1,
        reduce_lr_phase1
    ]
)

# ============================================================
# PHASE 2
# FINE-TUNE LAST PART OF EFFICIENTNET
# ============================================================

print()
print("=" * 70)
print("PHASE 2 - EFFICIENTNET FINE-TUNING")
print("=" * 70)

efficientnet.trainable = True

# Freeze most of EfficientNet
# Only the final ~30 layers will be fine-tuned.
total_layers = len(efficientnet.layers)

freeze_until = max(0, total_layers - 30)

for i, layer in enumerate(efficientnet.layers):

    if i < freeze_until:
        layer.trainable = False
    else:
        layer.trainable = True

# Keep BatchNormalization frozen for stable fine-tuning
for layer in efficientnet.layers:

    if isinstance(layer, layers.BatchNormalization):
        layer.trainable = False

trainable_count = sum(
    1 for layer in model.layers
    if layer.trainable
)

print()
print("EfficientNetB0:")
print(f"Total internal layers : {total_layers}")
print(f"Frozen until layer    : {freeze_until}")
print("Final layers          : TRAINABLE")
print("BatchNormalization    : FROZEN")

print()
print("Trainable top-level layers:", trainable_count)

# ============================================================
# COMPILE PHASE 2
# ============================================================

model.compile(
    optimizer=keras.optimizers.Adam(
        learning_rate=1e-5
    ),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# ============================================================
# CALLBACKS PHASE 2
# ============================================================

checkpoint_phase2 = keras.callbacks.ModelCheckpoint(
    str(OUTPUT_MODEL_PATH),
    monitor="val_accuracy",
    save_best_only=True,
    verbose=1
)

early_stop_phase2 = keras.callbacks.EarlyStopping(
    monitor="val_accuracy",
    patience=4,
    restore_best_weights=True,
    verbose=1
)

reduce_lr_phase2 = keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=2,
    min_lr=1e-7,
    verbose=1
)

# ============================================================
# TRAIN PHASE 2
# ============================================================

print()
print("=" * 70)
print("STARTING PHASE 2 FINE-TUNING")
print("=" * 70)

history2 = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=10,
    callbacks=[
        checkpoint_phase2,
        early_stop_phase2,
        reduce_lr_phase2
    ]
)

# ============================================================
# SAVE FINAL MODEL
# ============================================================

print()
print("=" * 70)
print("SAVING FINAL MOBILE-ADAPTED MODEL")
print("=" * 70)

model.save(OUTPUT_MODEL_PATH)

print()
print("Model saved successfully:")
print(OUTPUT_MODEL_PATH)

# ============================================================
# FINAL SUMMARY
# ============================================================

print()
print("=" * 70)
print("TRAINING COMPLETE")
print("=" * 70)

print()
print("Base model:")
print(BASE_MODEL_PATH)

print()
print("New mobile-adapted model:")
print(OUTPUT_MODEL_PATH)

print()
print("Classes:")
for i, name in enumerate(CLASS_NAMES):
    print(f"{i}: {name}")

print()
print("Next step:")
print("Evaluate the new model on the original internal test dataset")
print("and the external IMD2020 dataset.")

print("=" * 70)