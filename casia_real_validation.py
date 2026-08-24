import tensorflow as tf
import numpy as np
from pathlib import Path
from PIL import Image
from collections import Counter

# ============================================================
# TRUTHLENS - CASIA 2.0 REAL VALIDATION
# ============================================================

MODEL_PATH = Path(
    r"D:\TruthLens\Models\truthlens_model.tflite"
)

TEST_DIR = Path(
    r"D:\TruthLens\RealWorld_CASIA_Test\Real"
)

RESULT_FILE = Path(
    r"D:\TruthLens\Results\casia_real_validation_results.txt"
)

CLASS_NAMES = [
    "AI_Generated",
    "Deepfake",
    "Manipulated",
    "Real"
]

IMAGE_SIZE = (224, 224)

EXPECTED_CLASS = "Real"

# ============================================================
# HEADER
# ============================================================

print("=" * 70)
print("TRUTHLENS - CASIA 2.0 REAL VALIDATION")
print("=" * 70)

print("\nModel:")
print(MODEL_PATH)

print("\nTest dataset:")
print(TEST_DIR)

# ============================================================
# CHECK DATASET
# ============================================================

images = []

extensions = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff"
}

for file in TEST_DIR.iterdir():

    if file.is_file() and file.suffix.lower() in extensions:
        images.append(file)

images.sort()

print(f"\nImages found: {len(images)}")

if len(images) == 0:
    raise ValueError("No images found.")

# ============================================================
# LOAD TFLITE MODEL
# ============================================================

print("\n" + "=" * 70)
print("LOADING TFLITE MODEL")
print("=" * 70)

interpreter = tf.lite.Interpreter(
    model_path=str(MODEL_PATH)
)

interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print("\nInput shape:")
print(input_details[0]["shape"])

print("\nInput dtype:")
print(input_details[0]["dtype"])

print("\nOutput shape:")
print(output_details[0]["shape"])

print("\nOutput dtype:")
print(output_details[0]["dtype"])

# ============================================================
# PREPROCESS
# ============================================================

def preprocess(image_path):

    image = Image.open(image_path).convert("RGB")

    image = image.resize(IMAGE_SIZE)

    image = np.array(
        image,
        dtype=np.float32
    )

    # IMPORTANT:
    # EfficientNetB0 model expects pixel values in 0-255
    # because preprocessing is included in the model.

    image = np.expand_dims(
        image,
        axis=0
    )

    return image


# ============================================================
# PREDICTION
# ============================================================

def predict(image):

    interpreter.set_tensor(
        input_details[0]["index"],
        image
    )

    interpreter.invoke()

    output = interpreter.get_tensor(
        output_details[0]["index"]
    )

    probabilities = output[0]

    predicted_index = np.argmax(
        probabilities
    )

    confidence = probabilities[
        predicted_index
    ]

    return (
        CLASS_NAMES[predicted_index],
        float(confidence)
    )


# ============================================================
# VALIDATION
# ============================================================

print("\n" + "=" * 70)
print("RUNNING CASIA REAL VALIDATION")
print("=" * 70)

correct = 0
total = 0

confidences = []

prediction_counter = Counter()

for index, image_path in enumerate(
    images,
    start=1
):

    image = preprocess(image_path)

    predicted_class, confidence = predict(
        image
    )

    prediction_counter[
        predicted_class
    ] += 1

    confidences.append(
        confidence
    )

    if predicted_class == EXPECTED_CLASS:
        correct += 1

    total += 1

    print(
        f"{index:03d}/{len(images)}  "
        f"{image_path.name:35} "
        f"-> {predicted_class:15} "
        f"{confidence * 100:6.2f}%"
    )


# ============================================================
# RESULTS
# ============================================================

accuracy = (
    correct / total
) * 100

average_confidence = (
    np.mean(confidences)
) * 100

print("\n" + "=" * 70)
print("CASIA REAL VALIDATION RESULTS")
print("=" * 70)

print(
    f"\nTotal images tested : {total}"
)

print(
    f"Expected class      : {EXPECTED_CLASS}"
)

print(
    f"Correctly classified: {correct}"
)

print(
    f"Accuracy            : {accuracy:.2f}%"
)

print(
    f"Average confidence  : "
    f"{average_confidence:.2f}%"
)

print("\n" + "=" * 70)
print("PREDICTION DISTRIBUTION")
print("=" * 70)

for class_name in CLASS_NAMES:

    count = prediction_counter[
        class_name
    ]

    percentage = (
        count / total
    ) * 100

    print(
        f"{class_name:15}: "
        f"{count:3d} / {total} "
        f"({percentage:6.2f}%)"
    )


# ============================================================
# SAVE RESULTS
# ============================================================

RESULT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True
)

with open(
    RESULT_FILE,
    "w",
    encoding="utf-8"
) as file:

    file.write(
        "TruthLens CASIA 2.0 Real Validation\n"
    )

    file.write(
        "====================================\n\n"
    )

    file.write(
        f"Model: {MODEL_PATH}\n"
    )

    file.write(
        f"Dataset: {TEST_DIR}\n\n"
    )

    file.write(
        f"Total Images: {total}\n"
    )

    file.write(
        f"Expected Class: {EXPECTED_CLASS}\n"
    )

    file.write(
        f"Correct: {correct}\n"
    )

    file.write(
        f"Accuracy: {accuracy:.2f}%\n"
    )

    file.write(
        f"Average Confidence: "
        f"{average_confidence:.2f}%\n\n"
    )

    file.write(
        "Prediction Distribution:\n"
    )

    for class_name in CLASS_NAMES:

        count = prediction_counter[
            class_name
        ]

        percentage = (
            count / total
        ) * 100

        file.write(
            f"{class_name}: "
            f"{count}/{total} "
            f"({percentage:.2f}%)\n"
        )

print("\n" + "=" * 70)
print("CASIA REAL VALIDATION COMPLETE")
print("=" * 70)

print("\nResults saved to:")
print(RESULT_FILE)