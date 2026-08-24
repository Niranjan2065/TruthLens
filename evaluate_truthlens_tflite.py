import os
from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score
)

import matplotlib.pyplot as plt


# ============================================================
# TRUTHLENS - FINAL TFLITE MODEL EVALUATION
# ============================================================

PROJECT_DIR = Path(r"D:\TruthLens")

MODEL_PATH = (
    PROJECT_DIR
    / "Models"
    / "truthlens_model.tflite"
)

TEST_DIR = (
    PROJECT_DIR
    / "Final_Dataset"
    / "Test"
)

RESULTS_DIR = (
    PROJECT_DIR
    / "Results"
)

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# CONFIGURATION
# ============================================================

IMAGE_SIZE = (224, 224)

CLASS_NAMES = [
    "AI_Generated",
    "Deepfake",
    "Manipulated",
    "Real"
]


# ============================================================
# HEADER
# ============================================================

print("=" * 70)
print("TRUTHLENS - FINAL TFLITE MODEL EVALUATION")
print("=" * 70)

print("\nModel:")
print(MODEL_PATH)

print("\nTest dataset:")
print(TEST_DIR)


# ============================================================
# CHECK FILES
# ============================================================

if not MODEL_PATH.exists():

    raise FileNotFoundError(
        f"TFLite model not found:\n{MODEL_PATH}"
    )


if not TEST_DIR.exists():

    raise FileNotFoundError(
        f"Test dataset not found:\n{TEST_DIR}"
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

input_details = (
    interpreter.get_input_details()
)

output_details = (
    interpreter.get_output_details()
)

print("\nInput shape:")
print(input_details[0]["shape"])

print("\nInput dtype:")
print(input_details[0]["dtype"])

print("\nOutput shape:")
print(output_details[0]["shape"])

print("\nOutput dtype:")
print(output_details[0]["dtype"])


# ============================================================
# VERIFY MODEL INPUT
# ============================================================

input_shape = input_details[0]["shape"]

if (
    input_shape[1] != 224
    or input_shape[2] != 224
    or input_shape[3] != 3
):

    raise ValueError(
        f"Unexpected model input shape: {input_shape}"
    )


# ============================================================
# IMAGE PREPROCESSING
# ============================================================

def preprocess_image(image_path):

    image = Image.open(
        image_path
    ).convert("RGB")

    image = image.resize(
        IMAGE_SIZE
    )

    # IMPORTANT:
    #
    # EfficientNetB0 training pipeline did NOT
    # manually divide by 255.
    #
    # Therefore keep pixels in 0-255 range.
    #
    image = np.asarray(
        image,
        dtype=np.float32
    )

    image = np.expand_dims(
        image,
        axis=0
    )

    return image


# ============================================================
# PREDICTION
# ============================================================

def predict(image_path):

    image = preprocess_image(
        image_path
    )

    interpreter.set_tensor(
        input_details[0]["index"],
        image
    )

    interpreter.invoke()

    output = interpreter.get_tensor(
        output_details[0]["index"]
    )[0]

    predicted_index = int(
        np.argmax(output)
    )

    confidence = float(
        output[predicted_index]
    )

    return predicted_index, confidence, output


# ============================================================
# COLLECT DATA
# ============================================================

true_labels = []

predicted_labels = []

all_confidences = []

class_counts = {
    class_name: 0
    for class_name in CLASS_NAMES
}

correct_counts = {
    class_name: 0
    for class_name in CLASS_NAMES
}


# ============================================================
# RUN EVALUATION
# ============================================================

print("\n" + "=" * 70)
print("RUNNING FULL TEST DATASET")
print("=" * 70)


for class_index, class_name in enumerate(CLASS_NAMES):

    class_folder = (
        TEST_DIR / class_name
    )

    if not class_folder.exists():

        print(
            f"\nWARNING: Missing folder: "
            f"{class_folder}"
        )

        continue

    image_files = []

    for extension in [
        "*.jpg",
        "*.jpeg",
        "*.png",
        "*.bmp",
        "*.webp"
    ]:

        image_files.extend(
            class_folder.glob(extension)
        )

    print(
        f"\n{class_name}: "
        f"{len(image_files)} images"
    )

    for image_path in image_files:

        try:

            predicted_index, confidence, output = (
                predict(image_path)
            )

            true_labels.append(
                class_index
            )

            predicted_labels.append(
                predicted_index
            )

            all_confidences.append(
                confidence
            )

            class_counts[
                class_name
            ] += 1

            if predicted_index == class_index:

                correct_counts[
                    class_name
                ] += 1

        except Exception as error:

            print(
                f"\nERROR processing:"
                f" {image_path.name}"
            )

            print(error)


# ============================================================
# CONVERT TO NUMPY
# ============================================================

true_labels = np.array(
    true_labels
)

predicted_labels = np.array(
    predicted_labels
)

all_confidences = np.array(
    all_confidences
)


# ============================================================
# CHECK RESULTS
# ============================================================

total_images = len(
    true_labels
)

if total_images == 0:

    raise RuntimeError(
        "No images were successfully evaluated."
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
    average="weighted",
    zero_division=0
)

recall = recall_score(
    true_labels,
    predicted_labels,
    average="weighted",
    zero_division=0
)

f1 = f1_score(
    true_labels,
    predicted_labels,
    average="weighted",
    zero_division=0
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


# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    true_labels,
    predicted_labels,
    labels=range(len(CLASS_NAMES))
)


# ============================================================
# PRINT FINAL RESULTS
# ============================================================

print("\n")
print("=" * 70)
print("FINAL TFLITE RESULTS")
print("=" * 70)

print(
    f"\nTotal images tested : "
    f"{total_images}"
)

print(
    f"Accuracy            : "
    f"{accuracy * 100:.2f}%"
)

print(
    f"Precision           : "
    f"{precision * 100:.2f}%"
)

print(
    f"Recall              : "
    f"{recall * 100:.2f}%"
)

print(
    f"F1 Score            : "
    f"{f1 * 100:.2f}%"
)

print(
    f"Average confidence  : "
    f"{np.mean(all_confidences) * 100:.2f}%"
)


# ============================================================
# PER-CLASS ACCURACY
# ============================================================

print("\n" + "=" * 70)
print("PER-CLASS ACCURACY")
print("=" * 70)

for class_name in CLASS_NAMES:

    total = class_counts[
        class_name
    ]

    correct = correct_counts[
        class_name
    ]

    if total > 0:

        class_accuracy = (
            correct / total
        ) * 100

        print(
            f"{class_name:15s}: "
            f"{correct:4d}/{total:4d} "
            f"({class_accuracy:.2f}%)"
        )


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

print("\n" + "=" * 70)
print("CLASSIFICATION REPORT")
print("=" * 70)

print(
    "\n" + report
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

print("\n" + "=" * 70)
print("CONFUSION MATRIX")
print("=" * 70)

print()

print(
    "Rows    = Actual"
)

print(
    "Columns = Predicted"
)

print()

print(
    "              "
    + " ".join(
        f"{name[:10]:>12}"
        for name in CLASS_NAMES
    )
)

for i, row in enumerate(cm):

    print(
        f"{CLASS_NAMES[i]:15s}"
        + " ".join(
            f"{value:12d}"
            for value in row
        )
    )


# ============================================================
# SAVE TEXT RESULTS
# ============================================================

results_file = (
    RESULTS_DIR
    / "final_tflite_evaluation.txt"
)

with open(
    results_file,
    "w",
    encoding="utf-8"
) as file:

    file.write(
        "TruthLens Final TFLite Evaluation\n"
    )

    file.write(
        "=================================\n\n"
    )

    file.write(
        f"Model: {MODEL_PATH}\n"
    )

    file.write(
        f"Test Dataset: {TEST_DIR}\n\n"
    )

    file.write(
        f"Total Images: {total_images}\n"
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
        f"{np.mean(all_confidences) * 100:.4f}%\n\n"
    )

    file.write(
        "Class Mapping:\n"
    )

    for index, name in enumerate(CLASS_NAMES):

        file.write(
            f"{index} -> {name}\n"
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
        np.array2string(cm)
    )


# ============================================================
# SAVE CONFUSION MATRIX IMAGE
# ============================================================

plt.figure(
    figsize=(8, 7)
)

plt.imshow(
    cm,
    interpolation="nearest"
)

plt.title(
    "TruthLens TFLite Confusion Matrix"
)

plt.colorbar()

tick_marks = np.arange(
    len(CLASS_NAMES)
)

plt.xticks(
    tick_marks,
    CLASS_NAMES,
    rotation=45,
    ha="right"
)

plt.yticks(
    tick_marks,
    CLASS_NAMES
)

plt.xlabel(
    "Predicted Class"
)

plt.ylabel(
    "Actual Class"
)

# Display values inside cells
for i in range(
    len(CLASS_NAMES)
):

    for j in range(
        len(CLASS_NAMES)
    ):

        plt.text(
            j,
            i,
            str(cm[i, j]),
            ha="center",
            va="center"
        )


plt.tight_layout()

confusion_path = (
    RESULTS_DIR
    / "final_tflite_confusion_matrix.png"
)

plt.savefig(
    confusion_path,
    dpi=200,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# COMPLETE
# ============================================================

print("\n" + "=" * 70)
print("EVALUATION COMPLETE")
print("=" * 70)

print(
    "\nText results saved to:"
)

print(results_file)

print(
    "\nConfusion matrix saved to:"
)

print(confusion_path)

print("\n")