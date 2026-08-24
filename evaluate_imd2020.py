from pathlib import Path
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

# ============================================================
# TRUTHLENS - IMD2020 EXTERNAL VALIDATION
# ============================================================

PROJECT_DIR = Path(r"D:\TruthLens")

MODEL_PATH = (
    PROJECT_DIR
    / "Models"
    / "truthlens_efficientnetb0_finetuned_final.keras"
)

DATASET_DIR = PROJECT_DIR / "External_Test_IMD2020"

REAL_DIR = DATASET_DIR / "Real"
MANIPULATED_DIR = DATASET_DIR / "Manipulated"

RESULTS_DIR = PROJECT_DIR / "Results"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

RESULT_FILE = RESULTS_DIR / "imd2020_validation_results.txt"
CONFUSION_FILE = RESULTS_DIR / "imd2020_confusion_matrix.txt"

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
print("TRUTHLENS - IMD2020 EXTERNAL VALIDATION")
print("=" * 70)

print("\nModel:")
print(MODEL_PATH)

print("\nDataset:")
print(DATASET_DIR)

# ============================================================
# CHECK PATHS
# ============================================================

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Model not found:\n{MODEL_PATH}"
    )

if not REAL_DIR.exists():
    raise FileNotFoundError(
        f"Real directory not found:\n{REAL_DIR}"
    )

if not MANIPULATED_DIR.exists():
    raise FileNotFoundError(
        f"Manipulated directory not found:\n{MANIPULATED_DIR}"
    )

# ============================================================
# FIND IMAGES
# ============================================================

extensions = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff"
}

real_images = sorted([
    p for p in REAL_DIR.iterdir()
    if p.is_file() and p.suffix.lower() in extensions
])

manipulated_images = sorted([
    p for p in MANIPULATED_DIR.iterdir()
    if p.is_file() and p.suffix.lower() in extensions
])

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

model = keras.models.load_model(MODEL_PATH)

print("\nModel loaded successfully.")

print("\nInput shape:")
print(model.input_shape)

print("\nOutput shape:")
print(model.output_shape)

# ============================================================
# PREDICTION FUNCTION
# ============================================================

def predict_image(image_path):

    image = keras.utils.load_img(
        image_path,
        target_size=IMAGE_SIZE
    )

    image_array = keras.utils.img_to_array(image)

    image_array = np.expand_dims(
        image_array,
        axis=0
    )

    predictions = model.predict(
        image_array,
        verbose=0
    )[0]

    predicted_class = int(
        np.argmax(predictions)
    )

    confidence = float(
        predictions[predicted_class]
    )

    return predicted_class, confidence


# ============================================================
# RUN VALIDATION
# ============================================================

print("\n" + "=" * 70)
print("RUNNING IMD2020 EXTERNAL VALIDATION")
print("=" * 70)

true_labels = []
predicted_labels = []
confidences = []

# ------------------------------------------------------------
# REAL
# ------------------------------------------------------------

print("\nTesting Real images...")

for index, image_path in enumerate(real_images, start=1):

    predicted_class, confidence = predict_image(
        image_path
    )

    true_labels.append(3)  # Real
    predicted_labels.append(predicted_class)
    confidences.append(confidence)

    print(
        f"{index:03d}/{len(real_images)} "
        f"{image_path.name:<35} -> "
        f"{CLASS_NAMES[predicted_class]:<15} "
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

    true_labels.append(2)  # Manipulated
    predicted_labels.append(predicted_class)
    confidences.append(confidence)

    print(
        f"{index:03d}/{len(manipulated_images)} "
        f"{image_path.name:<35} -> "
        f"{CLASS_NAMES[predicted_class]:<15} "
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

cm = confusion_matrix(
    true_labels,
    predicted_labels,
    labels=[2, 3]
)

# ============================================================
# PREDICTION DISTRIBUTION
# ============================================================

prediction_counts = np.bincount(
    predicted_labels,
    minlength=4
)

# ============================================================
# DISPLAY RESULTS
# ============================================================

print("\n" + "=" * 70)
print("IMD2020 EXTERNAL VALIDATION RESULTS")
print("=" * 70)

print(
    f"\nTotal images tested : "
    f"{len(true_labels)}"
)

print(
    f"Accuracy            : "
    f"{accuracy * 100:.2f}%"
)

print(
    f"Macro Precision     : "
    f"{precision * 100:.2f}%"
)

print(
    f"Macro Recall        : "
    f"{recall * 100:.2f}%"
)

print(
    f"Macro F1 Score      : "
    f"{f1 * 100:.2f}%"
)

print(
    f"Average Confidence  : "
    f"{average_confidence * 100:.2f}%"
)

print("\n" + "=" * 70)
print("PREDICTION DISTRIBUTION")
print("=" * 70)

for index, class_name in enumerate(CLASS_NAMES):

    count = prediction_counts[index]

    percentage = (
        count / len(predicted_labels)
    ) * 100

    print(
        f"{class_name:<15}: "
        f"{count:3d}/{len(predicted_labels)} "
        f"({percentage:6.2f}%)"
    )

print("\n" + "=" * 70)
print("CLASSIFICATION REPORT")
print("=" * 70)

print(report)

print("\n" + "=" * 70)
print("CONFUSION MATRIX")
print("=" * 70)

print("\nRows = Actual")
print("Columns = Predicted\n")

print(
    "                 Manipulated    Real"
)

print(
    f"Manipulated       "
    f"{cm[0][0]:8d}    "
    f"{cm[0][1]:8d}"
)

print(
    f"Real              "
    f"{cm[1][0]:8d}    "
    f"{cm[1][1]:8d}"
)

# ============================================================
# SAVE RESULTS
# ============================================================

with open(
    RESULT_FILE,
    "w",
    encoding="utf-8"
) as file:

    file.write(
        "TruthLens IMD2020 External Validation\n"
    )

    file.write(
        "=====================================\n\n"
    )

    file.write(
        f"Model: {MODEL_PATH}\n"
    )

    file.write(
        f"Dataset: {DATASET_DIR}\n\n"
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
        f"Accuracy: "
        f"{accuracy * 100:.4f}%\n"
    )

    file.write(
        f"Macro Precision: "
        f"{precision * 100:.4f}%\n"
    )

    file.write(
        f"Macro Recall: "
        f"{recall * 100:.4f}%\n"
    )

    file.write(
        f"Macro F1 Score: "
        f"{f1 * 100:.4f}%\n"
    )

    file.write(
        f"Average Confidence: "
        f"{average_confidence * 100:.4f}%\n\n"
    )

    file.write(
        "Prediction Distribution:\n"
    )

    for index, class_name in enumerate(CLASS_NAMES):

        count = prediction_counts[index]

        percentage = (
            count / len(predicted_labels)
        ) * 100

        file.write(
            f"{class_name}: "
            f"{count}/{len(predicted_labels)} "
            f"({percentage:.2f}%)\n"
        )

    file.write(
        "\nClassification Report:\n"
    )

    file.write(report)

    file.write(
        "\n\nConfusion Matrix:\n"
    )

    file.write(
        str(cm)
    )

with open(
    CONFUSION_FILE,
    "w",
    encoding="utf-8"
) as file:

    file.write(
        "IMD2020 Confusion Matrix\n"
    )

    file.write(
        "========================\n\n"
    )

    file.write(
        "Rows = Actual\n"
    )

    file.write(
        "Columns = Predicted\n\n"
    )

    file.write(
        "                 Manipulated    Real\n"
    )

    file.write(
        f"Manipulated       "
        f"{cm[0][0]:8d}    "
        f"{cm[0][1]:8d}\n"
    )

    file.write(
        f"Real              "
        f"{cm[1][0]:8d}    "
        f"{cm[1][1]:8d}\n"
    )

# ============================================================
# COMPLETE
# ============================================================

print("\n" + "=" * 70)
print("IMD2020 VALIDATION COMPLETE")
print("=" * 70)

print("\nResults saved to:")
print(RESULT_FILE)

print("\nConfusion matrix saved to:")
print(CONFUSION_FILE)

print("=" * 70)