import tensorflow as tf
from pathlib import Path

print("=" * 70)
print("TRUTHLENS - TENSORFLOW LITE CONVERTER")
print("=" * 70)

# --------------------------------------------------------
# Paths
# --------------------------------------------------------

MODEL_PATH = Path(r"D:\TruthLens\Models\truthlens_efficientnetb0_finetuned.keras")

OUTPUT_PATH = Path(r"D:\TruthLens\Models\truthlens_model.tflite")

# --------------------------------------------------------
# Check model
# --------------------------------------------------------

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model not found:\n{MODEL_PATH}")

print("\nLoading model...")

model = tf.keras.models.load_model(MODEL_PATH)

print("Model loaded successfully.")

# --------------------------------------------------------
# Convert
# --------------------------------------------------------

print("\nConverting to TensorFlow Lite...")

converter = tf.lite.TFLiteConverter.from_keras_model(model)

converter.optimizations = [
    tf.lite.Optimize.DEFAULT
]

tflite_model = converter.convert()

# --------------------------------------------------------
# Save
# --------------------------------------------------------

with open(OUTPUT_PATH, "wb") as f:
    f.write(tflite_model)

print("\nConversion completed.")

print("\nSaved to:")

print(OUTPUT_PATH)

size_mb = OUTPUT_PATH.stat().st_size / (1024 * 1024)

print(f"\nModel Size : {size_mb:.2f} MB")

print("\nSUCCESS")