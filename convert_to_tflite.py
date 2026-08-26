import os
import tensorflow as tf

MODEL_PATH = r"D:\TruthLens\Models\truthlens_mobile_adapted.keras"
OUTPUT_PATH = r"D:\TruthLens\Models\truthlens_mobile_adapted.tflite"

print("=" * 70)
print("TRUTHLENS - KERAS TO TFLITE CONVERSION")
print("=" * 70)

print("\nLoading model:")
print(MODEL_PATH)

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"Model not found:\n{MODEL_PATH}"
    )

model = tf.keras.models.load_model(MODEL_PATH)

print("\nModel loaded successfully.")
print("Input shape :", model.input_shape)
print("Output shape:", model.output_shape)

print("\nConverting to TensorFlow Lite...")

converter = tf.lite.TFLiteConverter.from_keras_model(model)

# Standard FP32 TFLite conversion.
# This is the safest first conversion for TruthLens.
converter.optimizations = []

tflite_model = converter.convert()

with open(OUTPUT_PATH, "wb") as f:
    f.write(tflite_model)

size_mb = os.path.getsize(OUTPUT_PATH) / (1024 * 1024)

print("\n" + "=" * 70)
print("CONVERSION COMPLETE")
print("=" * 70)

print(f"\nTFLite model:")
print(OUTPUT_PATH)

print(f"\nModel size: {size_mb:.2f} MB")

print("\nInput:")
print(model.input_shape)

print("\nOutput:")
print(model.output_shape)

print("\nNext step:")
print("Test the TFLite model before integrating it into Android.")