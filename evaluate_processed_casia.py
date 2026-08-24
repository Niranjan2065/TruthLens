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

# ============================================================
# TRUTHLENS - PREPROCESSED CASIA EVALUATION
# ============================================================

PROJECT_DIR = Path(r"D:\TruthLens")

MODEL_PATH = (
    PROJECT_DIR
    / "Models"
    / "truthlens_efficientnetb0_finetuned_final.keras"
)

DATASET_DIR = (
    PROJECT_DIR
    / "CASIA_Processed"
)

CLASS_NAMES = [
    "AI_Generated",
    "Deepfake",
    "Manipulated",
    "Real"
]

# External dataset contains only these two classes
DATASET_CLASSES = [
    "Real",
    "Manipulated"
]

IMAGE_SIZE = (224, 224)

# ============================================================
# HEADER
# ============================================================

print("=" * 70)
print("TRUTHLENS - PREPROCESSED CASIA KERAS EVALUATION")
print("=" * 70)

print("\nModel:")
print(MODEL_PATH)

print("\nDataset:")
print(DATASET_DIR)

# ============================================================
# LOAD MODEL
# ============================================================

print("\n" + "=" * 70)
print("LOADING KERAS MODEL")
print("=" * 70)

model = tf.keras.models.load_model(
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
        image
    ).astype(np.float32)

    # IMPORTANT:
    # EfficientNetB0 expects pixel values in 0-255
    # because preprocessing is included inside the model.

    image = np.expand_dims(
        image,
        axis=0
    )

    return image


# ============================================================
# PREDICTION
# ============================================================

def predict(image):

    prediction = model.predict(
        image,
        verbose=0
    )[0]

    class_id = int(
        np.argmax(prediction)
    )

    confidence = float(
        prediction[class_id]
    )

    return class_id, confidence


# ============================================================
# LOAD IMAGES
# ============================================================

all_true = []
all_pred = []
all_confidence = []

total_images = 0

print("\n" + "=" * 70)
print("SCANNING PREPROCESSED CASIA DATASET")
print("=" * 70)

for actual_class in DATASET_CLASSES:

    folder = (
        DATASET_DIR
        / actual_class
    )

    images = sorted(
        [
            f
            for f in folder.iterdir()
            if f.is_file()
            and f.suffix.lower()
            in {
                ".jpg",
                ".jpeg",
                ".png"
            }
        ]
    )

    print(
        f"\n{actual_class}: "
        f"{len(images)} images"
    )

    # Convert actual class to TruthLens class ID

    if actual_class == "Real":

        true_id = 3

    else:

        true_id = 2

    for image_path in images:

        try:

            image = preprocess_image(
                image_path
            )

            predicted_id, confidence = predict(
                image
            )

            all_true.append(
                true_id
            )

            all_pred.append(
                predicted_id
            )

            all_confidence.append(
                confidence
            )

            total_images += 1

        except Exception as e:

            print(
                f"ERROR: "
                f"{image_path.name}"
            )

            print(e)


# ============================================================
# RESULTS
# ============================================================

accuracy = accuracy_score(
    all_true,
    all_pred
)

precision = precision_score(
    all_true,
    all_pred,
    labels=[2, 3],
    average="macro",
    zero_division=0
)

recall = recall_score(
    all_true,
    all_pred,
    labels=[2, 3],
    average="macro",
    zero_division=0
)

f1 = f1_score(
    all_true,
    all_pred,
    labels=[2, 3],
    average="macro",
    zero_division=0
)

average_confidence = (
    np.mean(all_confidence)
)

# ============================================================
# HEADER
# ============================================================

print("\n" + "=" * 70)
print("PREPROCESSED CASIA RESULTS")
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

# ============================================================
# PREDICTION DISTRIBUTION
# ============================================================

print("\n" + "=" * 70)
print("PREDICTION DISTRIBUTION")
print("=" * 70)

for class_id, class_name in enumerate(
    CLASS_NAMES
):

    count = all_pred.count(
        class_id
    )

    percentage = (
        count / total_images
    ) * 100

    print(
        f"{class_name:<15}: "
        f"{count:3d}/{total_images} "
        f"({percentage:6.2f}%)"
    )

# ============================================================
# CLASSIFICATION REPORT
# ============================================================

print("\n" + "=" * 70)
print("CLASSIFICATION REPORT")
print("=" * 70)

print(
    classification_report(
        all_true,
        all_pred,
        labels=[2, 3],
        target_names=[
            "Manipulated",
            "Real"
        ],
        zero_division=0
    )
)

# ============================================================
# CONFUSION MATRIX
# ============================================================

print("\n" + "=" * 70)
print("CONFUSION MATRIX")
print("=" * 70)

cm = confusion_matrix(
    all_true,
    all_pred,
    labels=[2, 3]
)

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
    f"Manipulated       "
    f"{cm[0][0]:8d}"
    f"{cm[0][1]:12d}"
)

print(
    f"Real              "
    f"{cm[1][0]:8d}"
    f"{cm[1][1]:12d}"
)

# ============================================================
# SAVE RESULTS
# ============================================================

RESULT_FILE = (
    PROJECT_DIR
    / "Results"
    / "preprocessed_casia_results.txt"
)

with open(
    RESULT_FILE,
    "w",
    encoding="utf-8"
) as file:

    file.write(
        "TruthLens Preprocessed CASIA Evaluation\n"
    )

    file.write(
        "========================================\n\n"
    )

    file.write(
        f"Total Images: {total_images}\n"
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
        "Confusion Matrix:\n"
    )

    file.write(
        str(cm)
    )

print("\n" + "=" * 70)
print("EVALUATION COMPLETE")
print("=" * 70)

print("\nResults saved to:")
print(RESULT_FILE)