import tensorflow as tf
import numpy as np

from pathlib import Path
from PIL import Image

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

import matplotlib.pyplot as plt


# ============================================================
# TRUTHLENS - REAL-WORLD VALIDATION
# ============================================================

PROJECT_DIR = Path(r"D:\TruthLens")

MODEL_PATH = PROJECT_DIR / "Models" / "truthlens_model.tflite"

TEST_DIR = PROJECT_DIR / "RealWorld_Test"

RESULTS_DIR = PROJECT_DIR / "Results"

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# CLASS ORDER
# ============================================================

CLASS_NAMES = [
    "AI_Generated",
    "Deepfake",
    "Manipulated",
    "Real"
]

IMAGE_SIZE = (224, 224)


# ============================================================
# HEADER
# ============================================================

print("=" * 70)
print("TRUTHLENS - REAL-WORLD TFLITE VALIDATION")
print("=" * 70)

print("\nModel:")
print(MODEL_PATH)

print("\nReal-world dataset:")
print(TEST_DIR)


# ============================================================
# CHECK FILES
# ============================================================

if not MODEL_PATH.exists():

    raise FileNotFoundError(
        f"\nTFLite model not found:\n{MODEL_PATH}"
    )


if not TEST_DIR.exists():

    raise FileNotFoundError(
        f"\nReal-world dataset not found:\n{TEST_DIR}"
    )


for class_name in CLASS_NAMES:

    folder = TEST_DIR / class_name

    if not folder.exists():

        raise FileNotFoundError(
            f"\nClass folder not found:\n{folder}"
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

print("\nOutput dtype:")
print(output_details[0]["dtype"])


# ============================================================
# PREPROCESS IMAGE
# ============================================================

def preprocess_image(image_path):

    image = Image.open(image_path).convert("RGB")

    image = image.resize(IMAGE_SIZE)

    image = np.asarray(
        image,
        dtype=np.float32
    )

    # EfficientNetB0 model expects float32 RGB input.
    # The trained model contains its own preprocessing.

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

    predicted_index = int(
        np.argmax(probabilities)
    )

    confidence = float(
        probabilities[predicted_index]
    )

    return (
        predicted_index,
        confidence,
        probabilities
    )


# ============================================================
# COLLECT RESULTS
# ============================================================

true_labels = []

predicted_labels = []

confidences = []

total_images = 0


print("\n" + "=" * 70)
print("SCANNING REAL-WORLD DATASET")
print("=" * 70)


for class_index, class_name in enumerate(CLASS_NAMES):

    folder = TEST_DIR / class_name

    image_files = []

    for extension in [
        "*.jpg",
        "*.jpeg",
        "*.png",
        "*.bmp",
        "*.webp"
    ]:

        image_files.extend(
            folder.glob(extension)
        )

    print(
        f"\n{class_name}: "
        f"{len(image_files)} images"
    )

    for image_path in image_files:

        try:

            image = preprocess_image(
                image_path
            )

            predicted_index, confidence, probabilities = predict(
                image
            )

            true_labels.append(
                class_index
            )

            predicted_labels.append(
                predicted_index
            )

            confidences.append(
                confidence
            )

            total_images += 1

        except Exception as error:

            print(
                f"\nERROR: {image_path.name}"
            )

            print(error)


# ============================================================
# CHECK RESULTS
# ============================================================

if total_images == 0:

    raise RuntimeError(
        "\nNo images were found in the RealWorld_Test folders."
    )


# ============================================================
# METRICS
# ============================================================

accuracy = accuracy_score(
    true_labels,
    predicted_labels
)

precision = precision_score(
    true_labels,
    predicted_labels,
    average="macro",
    zero_division=0
)

recall = recall_score(
    true_labels,
    predicted_labels,
    average="macro",
    zero_division=0
)

f1 = f1_score(
    true_labels,
    predicted_labels,
    average="macro",
    zero_division=0
)

average_confidence = np.mean(
    confidences
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    true_labels,
    predicted_labels,
    labels=range(len(CLASS_NAMES))
)


# ============================================================
# RESULTS
# ============================================================

print("\n" + "=" * 70)
print("REAL-WORLD VALIDATION RESULTS")
print("=" * 70)

print(
    f"\nTotal images tested : {total_images}"
)

print(
    f"Accuracy            : {accuracy * 100:.2f}%"
)

print(
    f"Precision           : {precision * 100:.2f}%"
)

print(
    f"Recall              : {recall * 100:.2f}%"
)

print(
    f"F1 Score            : {f1 * 100:.2f}%"
)

print(
    f"Average confidence  : "
    f"{average_confidence * 100:.2f}%"
)


# ============================================================
# PER-CLASS ACCURACY
# ============================================================

print("\n" + "=" * 70)
print("PER-CLASS ACCURACY")
print("=" * 70)

for index, class_name in enumerate(CLASS_NAMES):

    actual_count = np.sum(
        np.array(true_labels) == index
    )

    correct_count = np.sum(
        (
            np.array(true_labels) == index
        )
        &
        (
            np.array(predicted_labels) == index
        )
    )

    if actual_count > 0:

        class_accuracy = (
            correct_count /
            actual_count
        ) * 100

    else:

        class_accuracy = 0

    print(
        f"{class_name:15} : "
        f"{correct_count}/{actual_count} "
        f"({class_accuracy:.2f}%)"
    )


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

report = classification_report(
    true_labels,
    predicted_labels,
    target_names=CLASS_NAMES,
    digits=4,
    zero_division=0
)

print("\n" + "=" * 70)
print("CLASSIFICATION REPORT")
print("=" * 70)

print(report)


# ============================================================
# CONFUSION MATRIX TEXT
# ============================================================

print("\n" + "=" * 70)
print("CONFUSION MATRIX")
print("=" * 70)

print("\nRows = Actual")
print("Columns = Predicted\n")

print(
    f"{'':15}"
    f"{'AI_Generat':>12}"
    f"{'Deepfake':>12}"
    f"{'Manipulate':>12}"
    f"{'Real':>12}"
)

for i, class_name in enumerate(CLASS_NAMES):

    print(
        f"{class_name:15}"
        f"{cm[i][0]:12}"
        f"{cm[i][1]:12}"
        f"{cm[i][2]:12}"
        f"{cm[i][3]:12}"
    )


# ============================================================
# SAVE TEXT RESULTS
# ============================================================

result_file = (
    RESULTS_DIR /
    "realworld_validation_results.txt"
)


with open(
    result_file,
    "w",
    encoding="utf-8"
) as file:

    file.write(
        "TruthLens Real-World TFLite Validation\n"
    )

    file.write(
        "=" * 60 + "\n\n"
    )

    file.write(
        f"Model: {MODEL_PATH}\n"
    )

    file.write(
        f"Dataset: {TEST_DIR}\n\n"
    )

    file.write(
        f"Total images tested: {total_images}\n"
    )

    file.write(
        f"Accuracy: {accuracy * 100:.4f}%\n"
    )

    file.write(
        f"Precision: {precision * 100:.4f}%\n"
    )

    file.write(
        f"Recall: {recall * 100:.4f}%\n"
    )

    file.write(
        f"F1 Score: {f1 * 100:.4f}%\n"
    )

    file.write(
        f"Average Confidence: "
        f"{average_confidence * 100:.4f}%\n\n"
    )

    file.write(
        "Class Mapping:\n"
    )

    for index, class_name in enumerate(CLASS_NAMES):

        file.write(
            f"{index} -> {class_name}\n"
        )

    file.write(
        "\nClassification Report:\n"
    )

    file.write(
        report
    )

    file.write(
        "\nConfusion Matrix:\n"
    )

    file.write(
        str(cm)
    )


# ============================================================
# SAVE CONFUSION MATRIX IMAGE
# ============================================================

plt.figure(
    figsize=(9, 7)
)

plt.imshow(
    cm
)

plt.title(
    "TruthLens Real-World Confusion Matrix"
)

plt.xlabel(
    "Predicted Class"
)

plt.ylabel(
    "Actual Class"
)

plt.xticks(
    range(len(CLASS_NAMES)),
    CLASS_NAMES,
    rotation=45,
    ha="right"
)

plt.yticks(
    range(len(CLASS_NAMES)),
    CLASS_NAMES
)


for i in range(len(CLASS_NAMES)):

    for j in range(len(CLASS_NAMES)):

        plt.text(
            j,
            i,
            cm[i, j],
            ha="center",
            va="center"
        )


plt.colorbar()

plt.tight_layout()


confusion_file = (
    RESULTS_DIR /
    "realworld_confusion_matrix.png"
)


plt.savefig(
    confusion_file,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# COMPLETE
# ============================================================

print("\n" + "=" * 70)
print("REAL-WORLD VALIDATION COMPLETE")
print("=" * 70)

print(
    "\nText results saved to:"
)

print(result_file)

print(
    "\nConfusion matrix saved to:"
)

print(confusion_file)

print(
    "\nNext step: compare these results with "
    "the 85.75% final TFLite benchmark."
)

print("=" * 70)