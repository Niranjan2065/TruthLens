import tensorflow as tf
import numpy as np
from pathlib import Path
from PIL import Image

# ============================================================
# TRUTHLENS - CASIA 2.0 VALIDATION
# Automatically detects expected class from folder name
# ============================================================

MODEL_PATH = Path(r"D:\TruthLens\Models\truthlens_model.tflite")

TEST_DIR = Path(r"D:\TruthLens\RealWorld_CASIA_Test")

IMG_SIZE = (224, 224)

CLASS_NAMES = [
    "AI_Generated",
    "Deepfake",
    "Manipulated",
    "Real"
]

# ============================================================
# SELECT FOLDER
# ============================================================

TEST_CLASS = "Real"

DATA_DIR = TEST_DIR / TEST_CLASS

if not DATA_DIR.exists():
    raise FileNotFoundError(
        f"Folder not found:\n{DATA_DIR}"
    )

EXPECTED_CLASS = TEST_CLASS

# ============================================================
# HEADER
# ============================================================

print("=" * 70)
print("TRUTHLENS - CASIA 2.0 VALIDATION")
print("=" * 70)

print(f"\nModel:")
print(MODEL_PATH)

print(f"\nTest dataset:")
print(DATA_DIR)

print(f"\nExpected class:")
print(EXPECTED_CLASS)


# ============================================================
# FIND UNIQUE IMAGES
# ============================================================

images = []

for path in DATA_DIR.iterdir():

    if path.is_file() and path.suffix.lower() in [
        ".jpg",
        ".jpeg",
        ".png"
    ]:
        images.append(path)

# Remove duplicates and sort
images = sorted(set(images))

# Use maximum 100 UNIQUE images
images = images[:100]

print(f"\nUnique images found: {len(images)}")
if len(images) == 0:
    raise FileNotFoundError(
        f"No images found in:\n{DATA_DIR}"
    )

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

# ============================================================
# PREPROCESS
# ============================================================

def preprocess(image_path):

    image = Image.open(image_path).convert("RGB")

    image = image.resize(IMG_SIZE)

    image = np.array(image).astype(np.float32)

    # EfficientNetB0 model expects float32 input.
    image = np.expand_dims(image, axis=0)

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

    class_id = int(np.argmax(output[0]))

    confidence = float(output[0][class_id])

    return CLASS_NAMES[class_id], confidence


# ============================================================
# VALIDATION
# ============================================================

print("\n" + "=" * 70)
print("RUNNING CASIA VALIDATION")
print("=" * 70)

correct = 0

prediction_counts = {
    "AI_Generated": 0,
    "Deepfake": 0,
    "Manipulated": 0,
    "Real": 0
}

confidence_values = []

for index, image_path in enumerate(images, start=1):

    image = preprocess(image_path)

    predicted_class, confidence = predict(image)

    prediction_counts[predicted_class] += 1

    confidence_values.append(confidence)

    if predicted_class == EXPECTED_CLASS:
        correct += 1

    print(
        f"{index:03d}/{len(images)} "
        f"{image_path.name:35} -> "
        f"{predicted_class:15} "
        f"{confidence * 100:6.2f}%"
    )

# ============================================================
# RESULTS
# ============================================================

total = len(images)

accuracy = correct / total

average_confidence = np.mean(confidence_values)

print("\n" + "=" * 70)
print("CASIA VALIDATION RESULTS")
print("=" * 70)

print(f"\nExpected class       : {EXPECTED_CLASS}")
print(f"Total images tested  : {total}")
print(f"Correctly classified : {correct}")
print(f"Accuracy             : {accuracy * 100:.2f}%")
print(
    f"Average confidence  : "
    f"{average_confidence * 100:.2f}%"
)

print("\n" + "=" * 70)
print("PREDICTION DISTRIBUTION")
print("=" * 70)

for class_name in CLASS_NAMES:

    count = prediction_counts[class_name]

    percentage = count / total * 100

    print(
        f"{class_name:15} : "
        f"{count:3d} / {total} "
        f"({percentage:6.2f}%)"
    )

# ============================================================
# SAVE RESULTS
# ============================================================

RESULTS_DIR = Path(r"D:\TruthLens\Results")

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)

result_file = (
    RESULTS_DIR /
    "casia_real_validation_results.txt"
)

with open(
    result_file,
    "w",
    encoding="utf-8"
) as file:

    file.write(
        "TruthLens CASIA 2.0 Real Validation\n"
    )

    file.write(
        "===================================\n\n"
    )

    file.write(
        f"Model: {MODEL_PATH}\n"
    )

    file.write(
        f"Dataset: {DATA_DIR}\n\n"
    )

    file.write(
        f"Expected Class: {EXPECTED_CLASS}\n"
    )

    file.write(
        f"Total Images: {total}\n"
    )

    file.write(
        f"Correct: {correct}\n"
    )

    file.write(
        f"Accuracy: {accuracy * 100:.2f}%\n"
    )

    file.write(
        f"Average Confidence: "
        f"{average_confidence * 100:.2f}%\n\n"
    )

    file.write(
        "Prediction Distribution:\n"
    )

    for class_name in CLASS_NAMES:

        count = prediction_counts[class_name]

        percentage = count / total * 100

        file.write(
            f"{class_name}: "
            f"{count}/{total} "
            f"({percentage:.2f}%)\n"
        )

print("\n" + "=" * 70)
print("CASIA VALIDATION COMPLETE")
print("=" * 70)

print("\nResults saved to:")
print(result_file)