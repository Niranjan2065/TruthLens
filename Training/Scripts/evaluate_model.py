import os
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow import keras

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
)

import matplotlib.pyplot as plt


# ============================================================
# TRUTHLENS - MODEL EVALUATION
# ============================================================

PROJECT_DIR = Path(r"D:\TruthLens")

MODEL_PATH = PROJECT_DIR / "Models" / "truthlens_efficientnetb0_finetuned.keras"
TEST_DIR = PROJECT_DIR / "Final_Dataset" / "Test"

RESULTS_DIR = PROJECT_DIR / "Results"

RESULTS_DIR.mkdir(exist_ok=True)


# ============================================================
# SETTINGS
# ============================================================

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 16


# ============================================================
# LOAD MODEL
# ============================================================

print("=" * 70)
print("TRUTHLENS MODEL EVALUATION")
print("=" * 70)

print("\nLoading model...")

model = keras.models.load_model(MODEL_PATH)

print("Model loaded successfully.")


# ============================================================
# LOAD TEST DATASET
# ============================================================

test_dataset = keras.utils.image_dataset_from_directory(
    TEST_DIR,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False,
)

class_names = test_dataset.class_names

print("\nClasses:")

for i, c in enumerate(class_names):
    print(f"{i} -> {c}")

test_dataset = test_dataset.prefetch(tf.data.AUTOTUNE)


# ============================================================
# PREDICTIONS
# ============================================================

print("\nRunning predictions...")

y_true = []
y_pred = []

for images, labels in test_dataset:

    predictions = model.predict(images, verbose=0)

    predicted_labels = np.argmax(predictions, axis=1)

    y_true.extend(labels.numpy())
    y_pred.extend(predicted_labels)

y_true = np.array(y_true)
y_pred = np.array(y_pred)


# ============================================================
# METRICS
# ============================================================

accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(
    y_true,
    y_pred,
    average="weighted"
)
recall = recall_score(
    y_true,
    y_pred,
    average="weighted"
)
f1 = f1_score(
    y_true,
    y_pred,
    average="weighted"
)

print("\n")
print("=" * 70)
print("OVERALL RESULTS")
print("=" * 70)

print(f"Accuracy : {accuracy*100:.2f}%")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

report = classification_report(
    y_true,
    y_pred,
    target_names=class_names,
    digits=4
)

print("\n")
print("=" * 70)
print("CLASSIFICATION REPORT")
print("=" * 70)

print(report)

report_file = RESULTS_DIR / "classification_report.txt"

with open(report_file, "w") as f:
    f.write(report)


# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(y_true, y_pred)

print("\nConfusion Matrix:\n")
print(cm)


# ============================================================
# SAVE CONFUSION MATRIX IMAGE
# ============================================================

plt.figure(figsize=(8, 8))

plt.imshow(cm)

plt.title("TruthLens Confusion Matrix")

plt.colorbar()

plt.xticks(
    np.arange(len(class_names)),
    class_names,
    rotation=45
)

plt.yticks(
    np.arange(len(class_names)),
    class_names
)

for i in range(len(class_names)):
    for j in range(len(class_names)):
        plt.text(
            j,
            i,
            cm[i, j],
            ha="center",
            va="center",
            color="white" if cm[i, j] > cm.max()/2 else "black"
        )

plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.tight_layout()

cm_file = RESULTS_DIR / "confusion_matrix.png"

plt.savefig(cm_file, dpi=300)

plt.close()


# ============================================================
# SAVE SUMMARY
# ============================================================

summary_file = RESULTS_DIR / "evaluation_summary.txt"

with open(summary_file, "w") as f:

    f.write("TruthLens Evaluation Summary\n")
    f.write("="*50 + "\n\n")

    f.write(f"Accuracy : {accuracy*100:.2f}%\n")
    f.write(f"Precision: {precision:.4f}\n")
    f.write(f"Recall   : {recall:.4f}\n")
    f.write(f"F1 Score : {f1:.4f}\n")


print("\n")
print("=" * 70)
print("FILES SAVED")
print("=" * 70)

print(report_file)
print(summary_file)
print(cm_file)

print("\nEvaluation completed successfully.")