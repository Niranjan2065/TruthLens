import tensorflow as tf
import numpy as np
from PIL import Image
from pathlib import Path

MODEL_PATH = r"D:\TruthLens\Models\truthlens_model.tflite"
TEST_DIR = Path(r"D:\TruthLens\Mobile_Test")

CLASS_NAMES = [
    "AI_Generated",
    "Deepfake",
    "Manipulated",
    "Real"
]

IMG_SIZE = (224, 224)


def predict_image(image_path, interpreter):

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    input_index = input_details[0]["index"]
    output_index = output_details[0]["index"]

    # Load image
    image = Image.open(image_path).convert("RGB")

    print(f"\nImage: {image_path.name}")
    print(f"Original size: {image.size}")

    # Same expected model size
    image = image.resize(IMG_SIZE)

    # IMPORTANT:
    # Your TFLite model expects FLOAT32.
    # Test the 0-1 preprocessing used in our verification.
    image_array = np.asarray(image, dtype=np.float32) / 255.0

    image_array = np.expand_dims(image_array, axis=0)

    interpreter.set_tensor(input_index, image_array)

    interpreter.invoke()

    output = interpreter.get_tensor(output_index)[0]

    # Softmax if necessary
    probabilities = tf.nn.softmax(output).numpy()

    predicted_index = int(np.argmax(probabilities))

    print("\nPredictions:")

    for i, class_name in enumerate(CLASS_NAMES):
        print(
            f"{class_name:15s}: "
            f"{probabilities[i] * 100:.2f}%"
        )

    print(
        f"\nPREDICTED: "
        f"{CLASS_NAMES[predicted_index]}"
    )

    print(
        f"CONFIDENCE: "
        f"{probabilities[predicted_index] * 100:.2f}%"
    )


print("=" * 70)
print("TRUTHLENS - EXACT MOBILE IMAGE TEST")
print("=" * 70)

print("\nModel:")
print(MODEL_PATH)

interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()

print("\nInput shape:")
print(input_details[0]["shape"])

print("\nInput dtype:")
print(input_details[0]["dtype"])

print("\nTesting images...")
print("=" * 70)

for category in ["Real", "AI_Generated"]:

    folder = TEST_DIR / category

    print(f"\n{'=' * 20} {category} {'=' * 20}")

    if not folder.exists():
        print("Folder not found:", folder)
        continue

    images = [
        x for x in folder.iterdir()
        if x.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp"]
    ]

    for image_path in images:
        predict_image(image_path, interpreter)

print("\n")
print("=" * 70)
print("TEST COMPLETE")
print("=" * 70)