import os
import numpy as np
import tensorflow as tf

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image_dataset_from_directory
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# ============================================================
# TRUTHLENS - MOBILE ADAPTED INTERNAL TEST EVALUATION
# ============================================================

MODEL_PATH = r"D:\TruthLens\Models\truthlens_mobile_adapted.keras"
TEST_DIR = r"D:\TruthLens\Final_Dataset\Test"
RESULT_PATH = r"D:\TruthLens\Results\mobile_adapted_internal_evaluation.txt"

IMG_SIZE = (224, 224)
BATCH_SIZE = 32

CLASS_NAMES = [
    "AI_Generated",
    "Deepfake",
    "Manipulated",
    "Real"
]

print("=" * 70)
print("TRUTHLENS - MOBILE ADAPTED INTERNAL TEST EVALUATION")
print("=" * 70)

print("\nModel:")
print(MODEL_PATH)

print("\nTest dataset:")
print(TEST_DIR)

# ============================================================
# CHECK FILES
# ============================================================

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model not found:\n{MODEL_PATH}")

if not os.path.exists(TEST_DIR):
    raise FileNotFoundError(f"Test dataset not found:\n{TEST_DIR}")

# ============================================================
# CHECK DATASET COUNTS
# ============================================================

print("\n" + "=" * 70)
print("CHECKING TEST DATASET")
print("=" * 70)

for class_name in CLASS_NAMES:

    class_dir = os.path.join(TEST_DIR, class_name)

    if not os.path.exists(class_dir):
        raise FileNotFoundError(
            f"Missing class folder:\n{class_dir}"
        )

    count = 0

    for filename in os.listdir(class_dir):
        filepath = os.path.join(class_dir, filename)

        if os.path.isfile(filepath):
            if filename.lower().endswith(
                (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
            ):
                count += 1

    print(f"{class_name:<15}: {count}")

# ============================================================
# LOAD DATASET
# ============================================================

print("\n" + "=" * 70)
print("LOADING TEST DATASET")
print("=" * 70)

test_ds = image_dataset_from_directory(
    TEST_DIR,
    labels="inferred",
    label_mode="int",
    class_names=CLASS_NAMES,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)

print("\nDataset class names:")
print(test_ds.class_names)

# ============================================================
# LOAD MODEL
# ============================================================

print("\n" + "=" * 70)
print("LOADING MOBILE-ADAPTED MODEL")
print("=" * 70)

model = load_model(MODEL_PATH)

print("Model loaded successfully.")

print("\nInput shape:")
print(model.input_shape)

print("\nOutput shape:")
print(model.output_shape)

# ============================================================
# PREDICTIONS
# ============================================================

print("\n" + "=" * 70)
print("RUNNING PREDICTIONS")
print("=" * 70)

y_true = []
y_pred = []

batch_number = 0

for images, labels in test_ds:

    batch_number += 1

    predictions = model.predict(
        images,
        verbose=0
    )

    predicted_classes = np.argmax(
        predictions,
        axis=1
    )

    y_true.extend(labels.numpy())
    y_pred.extend(predicted_classes)

    if batch_number % 10 == 0:
        print(f"Processed batches: {batch_number}")

y_true = np.array(y_true)
y_pred = np.array(y_pred)

# ============================================================
# ACCURACY
# ============================================================

accuracy = accuracy_score(
    y_true,
    y_pred
)

# ============================================================
# CLASSIFICATION REPORT
# ============================================================

report = classification_report(
    y_true,
    y_pred,
    target_names=CLASS_NAMES,
    digits=4
)

# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_true,
    y_pred
)

# ============================================================
# PREDICTION DISTRIBUTION
# ============================================================

prediction_counts = np.bincount(
    y_pred,
    minlength=len(CLASS_NAMES)
)

# ============================================================
# DISPLAY RESULTS
# ============================================================

print("\n" + "=" * 70)
print("MOBILE-ADAPTED MODEL RESULTS")
print("=" * 70)

print(f"\nTotal images tested : {len(y_true)}")
print(f"Accuracy            : {accuracy * 100:.2f}%")

print("\n" + "=" * 70)
print("CLASSIFICATION REPORT")
print("=" * 70)

print(report)

print("=" * 70)
print("CONFUSION MATRIX")
print("=" * 70)

print("\nRows = Actual")
print("Columns = Predicted\n")

print(
    f"{'':18}"
    + "".join(f"{name:16}" for name in CLASS_NAMES)
)

for i, row in enumerate(cm):

    print(
        f"{CLASS_NAMES[i]:18}"
        + "".join(f"{value:16}" for value in row)
    )

print("\n" + "=" * 70)
print("PREDICTION DISTRIBUTION")
print("=" * 70)

for i, class_name in enumerate(CLASS_NAMES):

    count = prediction_counts[i]

    percentage = (
        count / len(y_pred) * 100
    )

    print(
        f"{class_name:<15}: "
        f"{count:5d} "
        f"({percentage:6.2f}%)"
    )

# ============================================================
# SAVE RESULTS
# ============================================================

os.makedirs(
    os.path.dirname(RESULT_PATH),
    exist_ok=True
)

with open(
    RESULT_PATH,
    "w",
    encoding="utf-8"
) as f:

    f.write("=" * 70 + "\n")
    f.write("TRUTHLENS - MOBILE ADAPTED INTERNAL TEST EVALUATION\n")
    f.write("=" * 70 + "\n\n")

    f.write(f"Model:\n{MODEL_PATH}\n\n")
    f.write(f"Test dataset:\n{TEST_DIR}\n\n")

    f.write("=" * 70 + "\n")
    f.write("RESULTS\n")
    f.write("=" * 70 + "\n\n")

    f.write(
        f"Total images tested : {len(y_true)}\n"
    )

    f.write(
        f"Accuracy            : {accuracy * 100:.2f}%\n\n"
    )

    f.write("=" * 70 + "\n")
    f.write("CLASSIFICATION REPORT\n")
    f.write("=" * 70 + "\n")

    f.write(report)

    f.write("\n" + "=" * 70 + "\n")
    f.write("CONFUSION MATRIX\n")
    f.write("=" * 70 + "\n\n")

    f.write(
        f"{'':18}"
        + "".join(f"{name:16}" for name in CLASS_NAMES)
        + "\n"
    )

    for i, row in enumerate(cm):

        f.write(
            f"{CLASS_NAMES[i]:18}"
            + "".join(f"{value:16}" for value in row)
            + "\n"
        )

    f.write("\n" + "=" * 70 + "\n")
    f.write("PREDICTION DISTRIBUTION\n")
    f.write("=" * 70 + "\n")

    for i, class_name in enumerate(CLASS_NAMES):

        count = prediction_counts[i]

        percentage = (
            count / len(y_pred) * 100
        )

        f.write(
            f"{class_name:<15}: "
            f"{count:5d} "
            f"({percentage:6.2f}%)\n"
        )

print("\n" + "=" * 70)
print("EVALUATION COMPLETE")
print("=" * 70)

print("\nResults saved to:")
print(RESULT_PATH)