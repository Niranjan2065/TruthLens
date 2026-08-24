import tensorflow as tf
import numpy as np
from pathlib import Path
from tensorflow import keras
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
# TRUTHLENS - KERAS MODEL CASIA EXTERNAL VALIDATION
# ============================================================

PROJECT_DIR = Path(r"D:\TruthLens")

MODEL_PATH = (
    PROJECT_DIR
    / "Models"
    / "truthlens_efficientnetb0_finetuned_final.keras"
)

REAL_DIR = (
    PROJECT_DIR
    / "RealWorld_CASIA_Test"
    / "Real"
)

MANIPULATED_DIR = (
    PROJECT_DIR
    / "RealWorld_CASIA_Test"
    / "Manipulated"
)

RESULTS_DIR = PROJECT_DIR / "Results"

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
print("TRUTHLENS - KERAS CASIA EXTERNAL VALIDATION")
print("=" * 70)

print("\nModel:")
print(MODEL_PATH)

print("\nReal dataset:")
print(REAL_DIR)

print("\nManipulated dataset:")
print(MANIPULATED_DIR)

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
# COLLECT IMAGES
# ============================================================

def get_images(folder):

    return sorted([
        p for p in folder.iterdir()
        if p.is_file()
        and p.suffix.lower() in [
            ".jpg",
            ".jpeg",
            ".png"
        ]
    ])


real_images = get_images(REAL_DIR)
manipulated_images = get_images(MANIPULATED_DIR)

print("\n" + "=" * 70)
print("DATASET")
print("=" * 70)

print(f"\nReal images        : {len(real_images)}")
print(f"Manipulated images : {len(manipulated_images)}")

# ============================================================
# PREPROCESS
# ============================================================

def preprocess(image_path):

    image = Image.open(image_path).convert("RGB")

    image = image.resize(IMAGE_SIZE)

    image = np.array(image).astype(np.float32)

    # Same preprocessing used by the TFLite test
    image = image / 255.0

    image = np.expand_dims(
        image,
        axis=0
    )

    return image

# ============================================================
# PREDICT
# ============================================================

def predict(image):

    prediction = model.predict(
        image,
        verbose=0
    )

    prediction = prediction[0]

    class_id = int(
        np.argmax(prediction)
    )

    confidence = float(
        prediction[class_id]
    )

    return class_id, confidence

# ============================================================
# RUN VALIDATION
# ============================================================

print("\n" + "=" * 70)
print("RUNNING KERAS CASIA VALIDATION")
print("=" * 70)

y_true = []
y_pred = []
confidences = []

# ------------------------------------------------------------
# REAL
# ------------------------------------------------------------

print("\nTesting Real images...")

for image_path in real_images:

    image = preprocess(image_path)

    predicted_class, confidence = predict(image)

    y_true.append(3)
    y_pred.append(predicted_class)
    confidences.append(confidence)

# ------------------------------------------------------------
# MANIPULATED
# ------------------------------------------------------------

print("Testing Manipulated images...")

for image_path in manipulated_images:

    image = preprocess(image_path)

    predicted_class, confidence = predict(image)

    y_true.append(2)
    y_pred.append(predicted_class)
    confidences.append(confidence)

# ============================================================
# METRICS
# ============================================================

accuracy = accuracy_score(
    y_true,
    y_pred
)

precision = precision_score(
    y_true,
    y_pred,
    average="macro",
    zero_division=0
)

recall = recall_score(
    y_true,
    y_pred,
    average="macro",
    zero_division=0
)

f1 = f1_score(
    y_true,
    y_pred,
    average="macro",
    zero_division=0
)

average_confidence = np.mean(
    confidences
)

# ============================================================
# RESULTS
# ============================================================

print("\n" + "=" * 70)
print("KERAS CASIA VALIDATION RESULTS")
print("=" * 70)

print(f"\nTotal images tested : {len(y_true)}")

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
# PER CLASS
# ============================================================

print("\n" + "=" * 70)
print("PREDICTION DISTRIBUTION")
print("=" * 70)

for class_id, class_name in enumerate(CLASS_NAMES):

    count = y_pred.count(class_id)

    print(
        f"{class_name:15} : "
        f"{count:3d}/{len(y_pred)} "
        f"({count / len(y_pred) * 100:6.2f}%)"
    )

# ============================================================
# CLASSIFICATION REPORT
# ============================================================

print("\n" + "=" * 70)
print("CLASSIFICATION REPORT")
print("=" * 70)

print(
    classification_report(
        y_true,
        y_pred,
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
    y_true,
    y_pred,
    labels=[2, 3]
)

print("\nRows = Actual")
print("Columns = Predicted")

print("\n                 Manipulated    Real")

print(
    f"Manipulated       {cm[0][0]:6d}      "
    f"{cm[0][1]:6d}"
)

print(
    f"Real              {cm[1][0]:6d}      "
    f"{cm[1][1]:6d}"
)

# ============================================================
# SAVE RESULTS
# ============================================================

RESULT_FILE = (
    RESULTS_DIR
    / "keras_casia_validation_results.txt"
)

with open(
    RESULT_FILE,
    "w",
    encoding="utf-8"
) as file:

    file.write(
        "TruthLens Keras CASIA External Validation\n"
    )

    file.write(
        "=========================================\n\n"
    )

    file.write(
        f"Model: {MODEL_PATH}\n"
    )

    file.write(
        f"Total Images: {len(y_true)}\n"
    )

    file.write(
        f"Accuracy: {accuracy * 100:.2f}%\n"
    )

    file.write(
        f"Macro Precision: {precision * 100:.2f}%\n"
    )

    file.write(
        f"Macro Recall: {recall * 100:.2f}%\n"
    )

    file.write(
        f"Macro F1 Score: {f1 * 100:.2f}%\n"
    )

    file.write(
        f"Average Confidence: "
        f"{average_confidence * 100:.2f}%\n"
    )

    file.write("\nPrediction Distribution:\n")

    for class_id, class_name in enumerate(CLASS_NAMES):

        count = y_pred.count(class_id)

        file.write(
            f"{class_name}: "
            f"{count}/{len(y_pred)} "
            f"({count / len(y_pred) * 100:.2f}%)\n"
        )

    file.write("\nConfusion Matrix:\n")
    file.write(str(cm))

print("\n" + "=" * 70)
print("KERAS VALIDATION COMPLETE")
print("=" * 70)

print("\nResults saved to:")
print(RESULT_FILE)