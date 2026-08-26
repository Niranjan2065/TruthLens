import os
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from PIL import Image

# ============================================================
# TRUTHLENS - MOBILE ADAPTED MODEL REAL-WORLD VALIDATION
# ============================================================

MODEL_PATH = r"D:\TruthLens\Models\truthlens_mobile_adapted.keras"
DATASET_PATH = r"D:\TruthLens\RealWorld_Test"

CLASS_NAMES = [
    "AI_Generated",
    "Deepfake",
    "Manipulated",
    "Real"
]

IMAGE_SIZE = (224, 224)

print("=" * 70)
print("TRUTHLENS - MOBILE ADAPTED REAL-WORLD VALIDATION")
print("=" * 70)

print()
print("Model:")
print(MODEL_PATH)

print()
print("Dataset:")
print(DATASET_PATH)

# ============================================================
# CHECK FILES
# ============================================================

if not os.path.exists(MODEL_PATH):
    print()
    print("ERROR: Model not found.")
    print(MODEL_PATH)
    raise SystemExit

if not os.path.exists(DATASET_PATH):
    print()
    print("ERROR: Dataset not found.")
    print(DATASET_PATH)
    raise SystemExit

# ============================================================
# LOAD MODEL
# ============================================================

print()
print("=" * 70)
print("LOADING MOBILE-ADAPTED MODEL")
print("=" * 70)

model = tf.keras.models.load_model(MODEL_PATH)

print("Model loaded successfully.")
print()
print("Input shape :", model.input_shape)
print("Output shape:", model.output_shape)

# ============================================================
# SCAN DATASET
# ============================================================

print()
print("=" * 70)
print("SCANNING REAL-WORLD DATASET")
print("=" * 70)

image_paths = []
true_labels = []

valid_extensions = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff"
}

for class_index, class_name in enumerate(CLASS_NAMES):

    class_dir = os.path.join(DATASET_PATH, class_name)

    if not os.path.isdir(class_dir):
        print()
        print("WARNING: Missing class folder:", class_name)
        continue

    count = 0

    for filename in os.listdir(class_dir):

        filepath = os.path.join(class_dir, filename)

        if not os.path.isfile(filepath):
            continue

        extension = os.path.splitext(filename)[1].lower()

        if extension not in valid_extensions:
            continue

        image_paths.append(filepath)
        true_labels.append(class_index)

        count += 1

    print(f"{class_name:<15}: {count} images")

print()
print("Total images:", len(image_paths))

if len(image_paths) == 0:
    print("ERROR: No images found.")
    raise SystemExit

# ============================================================
# LOAD IMAGES
# ============================================================

print()
print("=" * 70)
print("LOADING IMAGES")
print("=" * 70)

images = []
valid_labels = []

for i, (filepath, label) in enumerate(
    zip(image_paths, true_labels)
):

    try:

        image = Image.open(filepath).convert("RGB")

        image = image.resize(IMAGE_SIZE)

        image_array = np.asarray(
            image,
            dtype=np.float32
        )

        images.append(image_array)
        valid_labels.append(label)

    except Exception as e:

        print()
        print("ERROR loading:", filepath)
        print(e)

X = np.asarray(images, dtype=np.float32)
y_true = np.asarray(valid_labels, dtype=np.int32)

print()
print("Input shape:", X.shape)
print("Input dtype:", X.dtype)
print("Input min  :", X.min())
print("Input max  :", X.max())

# ============================================================
# PREDICTIONS
# ============================================================

print()
print("=" * 70)
print("RUNNING PREDICTIONS")
print("=" * 70)

predictions = model.predict(
    X,
    batch_size=32,
    verbose=1
)

y_pred = np.argmax(
    predictions,
    axis=1
)

# ============================================================
# OVERALL RESULTS
# ============================================================

accuracy = accuracy_score(
    y_true,
    y_pred
)

print()
print("=" * 70)
print("MOBILE-ADAPTED REAL-WORLD RESULTS")
print("=" * 70)

print()
print("Total images tested :", len(y_true))
print(f"Accuracy            : {accuracy * 100:.2f}%")

# ============================================================
# PER-CLASS ACCURACY
# ============================================================

print()
print("=" * 70)
print("PER-CLASS ACCURACY")
print("=" * 70)

for class_index, class_name in enumerate(CLASS_NAMES):

    mask = y_true == class_index

    total = np.sum(mask)

    if total == 0:
        continue

    correct = np.sum(
        y_pred[mask] == class_index
    )

    class_accuracy = (
        correct / total
    ) * 100

    print(
        f"{class_name:<15}: "
        f"{correct}/{total} "
        f"({class_accuracy:.2f}%)"
    )

# ============================================================
# CLASSIFICATION REPORT
# ============================================================

print()
print("=" * 70)
print("CLASSIFICATION REPORT")
print("=" * 70)

report = classification_report(
    y_true,
    y_pred,
    target_names=CLASS_NAMES,
    digits=4
)

print(report)

# ============================================================
# CONFUSION MATRIX
# ============================================================

print("=" * 70)
print("CONFUSION MATRIX")
print("=" * 70)

cm = confusion_matrix(
    y_true,
    y_pred
)

print()
print("Rows = Actual")
print("Columns = Predicted")
print()

print(
    f"{'':15}"
    f"{'AI_Generated':>15}"
    f"{'Deepfake':>12}"
    f"{'Manipulated':>15}"
    f"{'Real':>10}"
)

for i, class_name in enumerate(CLASS_NAMES):

    print(
        f"{class_name:<15}"
        f"{cm[i,0]:>15}"
        f"{cm[i,1]:>12}"
        f"{cm[i,2]:>15}"
        f"{cm[i,3]:>10}"
    )

# ============================================================
# PREDICTION DISTRIBUTION
# ============================================================

print()
print("=" * 70)
print("PREDICTION DISTRIBUTION")
print("=" * 70)

for class_index, class_name in enumerate(CLASS_NAMES):

    count = np.sum(
        y_pred == class_index
    )

    percentage = (
        count / len(y_pred)
    ) * 100

    print(
        f"{class_name:<15}: "
        f"{count:4d} "
        f"({percentage:6.2f}%)"
    )

# ============================================================
# SAVE RESULTS
# ============================================================

RESULTS_DIR = r"D:\TruthLens\Results"

os.makedirs(
    RESULTS_DIR,
    exist_ok=True
)

RESULT_FILE = os.path.join(
    RESULTS_DIR,
    "mobile_adapted_realworld_results.txt"
)

with open(
    RESULT_FILE,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "TRUTHLENS - MOBILE ADAPTED "
        "REAL-WORLD VALIDATION\n"
    )

    f.write("=" * 70 + "\n\n")

    f.write(
        f"Model: {MODEL_PATH}\n"
    )

    f.write(
        f"Dataset: {DATASET_PATH}\n\n"
    )

    f.write(
        f"Total images: {len(y_true)}\n"
    )

    f.write(
        f"Accuracy: {accuracy * 100:.2f}%\n\n"
    )

    f.write(
        "CLASSIFICATION REPORT\n"
    )

    f.write("-" * 70 + "\n")

    f.write(report)

    f.write("\n\n")

    f.write(
        "CONFUSION MATRIX\n"
    )

    f.write("-" * 70 + "\n")

    f.write(
        str(cm)
    )

print()
print("=" * 70)
print("VALIDATION COMPLETE")
print("=" * 70)

print()
print("Results saved to:")
print(RESULT_FILE)

print()
print("=" * 70)