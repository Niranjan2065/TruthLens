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


print("=" * 70)
print("TRUTHLENS - MOBILE IMAGE PREPROCESSING COMPARISON")
print("=" * 70)

print("\nModel:")
print(MODEL_PATH)

interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

input_index = input_details[0]["index"]
output_index = output_details[0]["index"]

print("\nINPUT")
print("-" * 70)
print("Shape :", input_details[0]["shape"])
print("Dtype :", input_details[0]["dtype"])
print("Quantization :", input_details[0]["quantization"])

print("\nOUTPUT")
print("-" * 70)
print("Shape :", output_details[0]["shape"])
print("Dtype :", output_details[0]["dtype"])
print("Quantization :", output_details[0]["quantization"])


def test_image(image_path):

    print("\n")
    print("=" * 70)
    print("IMAGE:", image_path.name)
    print("EXPECTED:", image_path.parent.name)
    print("=" * 70)

    image = Image.open(image_path).convert("RGB")

    print("Original size:", image.size)

    image = image.resize(IMG_SIZE)

    image_np = np.asarray(image, dtype=np.float32)

    # ---------------------------------------------------------
    # TEST 1: 0-255
    # ---------------------------------------------------------

    input_255 = np.expand_dims(image_np, axis=0)

    interpreter.set_tensor(input_index, input_255)
    interpreter.invoke()

    raw_255 = interpreter.get_tensor(output_index)[0]

    prob_255 = tf.nn.softmax(raw_255).numpy()

    print("\n")
    print("*************** 0-255 PREPROCESSING ***************")

    print("\nRaw output:")
    print(raw_255)

    print("\nProbabilities:")

    for i, name in enumerate(CLASS_NAMES):
        print(f"{name:15s}: {prob_255[i] * 100:.2f}%")

    index_255 = int(np.argmax(prob_255))

    print("\nPrediction:")
    print(CLASS_NAMES[index_255])

    print(f"Confidence: {prob_255[index_255] * 100:.2f}%")

    # ---------------------------------------------------------
    # TEST 2: 0-1
    # ---------------------------------------------------------

    input_01 = input_255 / 255.0

    interpreter.set_tensor(input_index, input_01)
    interpreter.invoke()

    raw_01 = interpreter.get_tensor(output_index)[0]

    prob_01 = tf.nn.softmax(raw_01).numpy()

    print("\n")
    print("*************** 0-1 PREPROCESSING ***************")

    print("\nRaw output:")
    print(raw_01)

    print("\nProbabilities:")

    for i, name in enumerate(CLASS_NAMES):
        print(f"{name:15s}: {prob_01[i] * 100:.2f}%")

    index_01 = int(np.argmax(prob_01))

    print("\nPrediction:")
    print(CLASS_NAMES[index_01])

    print(f"Confidence: {prob_01[index_01] * 100:.2f}%")


# -------------------------------------------------------------
# RUN
# -------------------------------------------------------------

for category in ["Real", "AI_Generated"]:

    folder = TEST_DIR / category

    if not folder.exists():
        print("\nFolder missing:", folder)
        continue

    images = [
        x for x in folder.iterdir()
        if x.suffix.lower() in [
            ".jpg",
            ".jpeg",
            ".png",
            ".bmp"
        ]
    ]

    for image_path in images:
        test_image(image_path)


print("\n")
print("=" * 70)
print("PREPROCESSING COMPARISON COMPLETE")
print("=" * 70)