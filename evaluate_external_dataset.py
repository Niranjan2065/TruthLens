import tensorflow as tf
import numpy as np

from tensorflow import keras
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


# ============================================================
# TRUTHLENS - EXTERNAL DATASET VALIDATION
# ============================================================

PROJECT_DIR = Path(r"D:\TruthLens")

MODEL_PATH = (
    PROJECT_DIR
    / "Models"
    / "truthlens_efficientnetb0_finetuned_final.keras"
)

EXTERNAL_DATASET = (
    PROJECT_DIR
    / "External_Test"
)

RESULTS_DIR = PROJECT_DIR / "Results"

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# SETTINGS
# ============================================================

IMAGE_SIZE = (224, 224)

CLASS_NAMES = [
    "Manipulated",
    "Real"
]

CLASS_TO_ID = {
    "Manipulated": 2,
    "Real": 3
}


# ============================================================
# HEADER
# ============================================================

print("=" * 70)
print("TRUTHLENS - EXTERNAL DATASET VALIDATION")
print("=" * 70)

print("\nModel:")
print(MODEL_PATH)

print("\nExternal dataset:")
print(EXTERNAL_DATASET)


# ============================================================
# CHECK DATASET
# ============================================================

real_dir = EXTERNAL_DATASET / "Real"
manipulated_dir = EXTERNAL_DATASET / "Manipulated"

if not real_dir.exists():
    raise FileNotFoundError(
        f"Real folder not found:\n{real_dir}"
    )

if not manipulated_dir.exists():
    raise FileNotFoundError(
        f"Manipulated folder not found:\n{manipulated_dir}"
    )


# ============================================================
# COLLECT IMAGES
# ============================================================

extensions = [
    "*.jpg",
    "*.jpeg",
    "*.png",
    "*.bmp",
    "*.webp"
]


def collect_images(folder):

    images = []

    for extension in extensions:
        images.extend(folder.glob(extension))

    return sorted(images)


real_images = collect_images(real_dir)
manipulated_images = collect_images(manipulated_dir)


print("\n" + "=" * 70)
print("EXTERNAL DATASET")
print("=" * 70)

print(f"\nReal images        : {len(real_images)}")
print(f"Manipulated images : {len(manipulated_images)}")


if len(real_images) == 0:
    raise ValueError("No Real images found.")

if len(manipulated_images) == 0:
    raise ValueError("No Manipulated images found.")


# ============================================================
# LOAD MODEL
# ============================================================

print("\n" + "=" * 70)
print("LOADING KERAS MODEL")
print("=" * 70)

model = keras.models.load_model(
    MODEL_PATH
)

print("\nModel loaded successfully.")

print("\nInput shape:")
print(model.input_shape)

print("\nOutput shape:")
print(model.output_shape)


# ============================================================
# PREPROCESS
# ============================================================

def preprocess_image(image_path):

    image = Image.open(
        image_path
    ).convert("RGB")

    image = image.resize(
        IMAGE_SIZE
    )

    image = np.asarray(
        image,
        dtype=np.float32
    )

    # IMPORTANT:
    # EfficientNetB0 model handles its own
    # input rescaling internally.
    #
    # Therefore DO NOT divide by 255 here.

    image = np.expand_dims(
        image,
        axis=0
    )

    return image


# ============================================================
# PREDICTION
# ============================================================

def predict_image(image_path):

    image = preprocess_image(
        image_path
    )

    prediction = model.predict(
        image,
        verbose=0
    )[0]

    predicted_class = int(
        np.argmax(prediction)
    )

    confidence = float(
        prediction[predicted_class]
    )

    return predicted_class, confidence


# ============================================================
# RUN VALIDATION
# ============================================================

print("\n" + "=" * 70)
print("RUNNING EXTERNAL VALIDATION")
print("=" * 70)


true_labels = []
predicted_labels = []
confidences = []


# ------------------------------------------------------------
# REAL
# ------------------------------------------------------------

print("\nTesting Real images...")

for index, image_path in enumerate(
    real_images,
    start=1
):

    predicted_class, confidence = predict_image(
        image_path
    )

    true_labels.append(
        CLASS_TO_ID["Real"]
    )

    predicted_labels.append(
        predicted_class
    )

    confidences.append(
        confidence
    )

    if index <= 10:

        print(
            f"{index:03d}  "
            f"{image_path.name:35} -> "
            f"{predicted_class} "
            f"{confidence * 100:.2f}%"
        )


# ------------------------------------------------------------
# MANIPULATED
# ------------------------------------------------------------

print("\nTesting Manipulated images...")

for index, image_path in enumerate(
    manipulated_images,
    start=1
):

    predicted_class, confidence = predict_image(
        image_path
    )

    true_labels.append(
        CLASS_TO_ID["Manipulated"]
    )

    predicted_labels.append(
        predicted_class
    )

    confidences.append(
        confidence
    )

    if index <= 10:

        print(
            f"{index:03d}  "
            f"{image_path.name:35} -> "
            f"{predicted_class} "
            f"{confidence * 100:.2f}%"
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
    labels=[2, 3],
    average="macro",
    zero_division=0
)

recall = recall_score(
    true_labels,
    predicted_labels,
    labels=[2, 3],
    average="macro",
    zero_division=0
)

f1 = f1_score(
    true_labels,
    predicted_labels,
    labels=[2, 3],
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
    labels=[2, 3]
)


# ============================================================
# RESULTS
# ============================================================

report = classification_report(
    true_labels,
    predicted_labels,
    labels=[2, 3],
    target_names=[
        "Manipulated",
        "Real"
    ],
    zero_division=0
)


print("\n" + "=" * 70)
print("EXTERNAL VALIDATION RESULTS")
print("=" * 70)

print(
    f"\nTotal images tested : {len(true_labels)}"
)

print(
    f"Accuracy            : {accuracy * 100:.2f}%"
)

print(
    f"Macro Precision     : {precision * 100:.2f}%"
)

print(
    f"Macro Recall        : {recall * 100:.2f}%"
)

print(
    f"Macro F1 Score      : {f1 * 100:.2f}%"
)

print(
    f"Average Confidence  : "
    f"{average_confidence * 100:.2f}%"
)


# ============================================================
# PER-CLASS RESULTS
# ============================================================

print("\n" + "=" * 70)
print("CLASSIFICATION REPORT")
print("=" * 70)

print(report)


# ============================================================
# CONFUSION MATRIX
# ============================================================

print("\n" + "=" * 70)
print("CONFUSION MATRIX")
print("=" * 70)

print(
    "\nRows = Actual"
)

print(
    "Columns = Predicted"
)

print(
    "\n                 Manipulated    Real"
)

print(
    f"Manipulated      "
    f"{cm[0][0]:10d}"
    f"{cm[0][1]:13d}"
)

print(
    f"Real             "
    f"{cm[1][0]:10d}"
    f"{cm[1][1]:13d}"
)


# ============================================================
# SAVE RESULTS
# ============================================================

result_file = (
    RESULTS_DIR
    / "external_validation_results.txt"
)


with open(
    result_file,
    "w",
    encoding="utf-8"
) as file:

    file.write(
        "TruthLens External Dataset Validation\n"
    )

    file.write(
        "=====================================\n\n"
    )

    file.write(
        f"Model: {MODEL_PATH}\n"
    )

    file.write(
        f"Dataset: {EXTERNAL_DATASET}\n\n"
    )

    file.write(
        f"Real Images: {len(real_images)}\n"
    )

    file.write(
        f"Manipulated Images: "
        f"{len(manipulated_images)}\n"
    )

    file.write(
        f"Total Images: {len(true_labels)}\n\n"
    )

    file.write(
        f"Accuracy: {accuracy * 100:.2f}%\n"
    )

    file.write(
        f"Macro Precision: "
        f"{precision * 100:.2f}%\n"
    )

    file.write(
        f"Macro Recall: "
        f"{recall * 100:.2f}%\n"
    )

    file.write(
        f"Macro F1 Score: "
        f"{f1 * 100:.2f}%\n"
    )

    file.write(
        f"Average Confidence: "
        f"{average_confidence * 100:.2f}%\n\n"
    )

    file.write(
        "Classification Report:\n"
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
# COMPLETE
# ============================================================

print("\n" + "=" * 70)
print("EXTERNAL VALIDATION COMPLETE")
print("=" * 70)

print("\nResults saved to:")
print(result_file)