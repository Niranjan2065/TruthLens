import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from pathlib import Path

# ============================================================
# TRUTHLENS - EFFICIENTNETB0 FINE TUNING
# ============================================================

PROJECT_DIR = Path(r"D:\TruthLens")

TRAIN_DIR = PROJECT_DIR / "Final_Dataset" / "Train"
VAL_DIR = PROJECT_DIR / "Final_Dataset" / "Validation"

MODEL_DIR = PROJECT_DIR / "Models"
RESULTS_DIR = PROJECT_DIR / "Results"

BEST_BASELINE_MODEL = MODEL_DIR / "truthlens_efficientnetb0_best.keras"

FINE_TUNED_MODEL = MODEL_DIR / "truthlens_efficientnetb0_finetuned.keras"

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS = 3
LEARNING_RATE = 1e-5
SEED = 42

# ============================================================
# LOAD DATASET
# ============================================================

train_ds = keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=True,
    seed=SEED,
)

val_ds = keras.utils.image_dataset_from_directory(
    VAL_DIR,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False,
)

AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.prefetch(AUTOTUNE)
val_ds = val_ds.prefetch(AUTOTUNE)

# ============================================================
# LOAD BASELINE MODEL
# ============================================================

print("=" * 70)
print("LOADING BASELINE MODEL")
print("=" * 70)

model = keras.models.load_model(BEST_BASELINE_MODEL)

print(model.summary())

# ============================================================
# UNFREEZE LAST PART OF EFFICIENTNET
# ============================================================

base_model = None

for layer in model.layers:
    if "efficientnet" in layer.name.lower():
        base_model = layer
        break

if base_model is None:
    raise ValueError("EfficientNet layer not found.")

print("\nUnfreezing last 30 layers...")

base_model.trainable = True

for layer in base_model.layers[:-30]:
    layer.trainable = False

# ============================================================
# COMPILE
# ============================================================

model.compile(
    optimizer=keras.optimizers.Adam(
        learning_rate=LEARNING_RATE
    ),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# ============================================================
# CALLBACKS
# ============================================================

callbacks = [

    keras.callbacks.ModelCheckpoint(
        filepath=str(FINE_TUNED_MODEL),
        monitor="val_accuracy",
        save_best_only=True,
        mode="max",
        verbose=1
    ),

    keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=3,
        restore_best_weights=True,
        verbose=1
    ),

    keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.2,
        patience=2,
        verbose=1
    ),

    keras.callbacks.CSVLogger(
        RESULTS_DIR / "fine_tune_log.csv"
    )
]

# ============================================================
# TRAIN
# ============================================================

print("=" * 70)
print("STARTING FINE TUNING")
print("=" * 70)

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=callbacks
)

# ============================================================
# SAVE FINAL MODEL
# ============================================================

model.save(
    MODEL_DIR / "truthlens_efficientnetb0_finetuned_final.keras"
)

print("\nFine tuning completed.")