import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# ============================================================
# TRUTHLENS - INTERNAL KERAS TEST DATASET EVALUATION
# ============================================================

MODEL_PATH = r"D:\TruthLens\Models\truthlens_efficientnetb0_finetuned_final.keras"
TEST_DIR = r"D:\TruthLens\Final_Dataset\Test"

IMG_SIZE = (224, 224)

CLASS_NAMES = [
    "AI_Generated",
    "Deepfake",
    "Manipulated",
    "Real"
]

VALID_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


print("=" * 70)
print("TRUTHLENS - INTERNAL KERAS MODEL EVALUATION")
print("=" * 70)

print("\nModel:")
print(MODEL_PATH)

print("\nTest dataset:")
print(TEST_DIR)


# ============================================================
# CHECK PATHS
# ============================================================

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError("Model not found.")

if not os.path.exists(TEST_DIR):
    raise FileNotFoundError("Test dataset not found.")


# ============================================================
# LOAD MODEL
# ============================================================

print("\n" + "=" * 70)
print("LOADING MODEL")
print("=" * 70)

model = tf.keras.models.load_model(MODEL_PATH)

print("Model loaded successfully.")

print("\nInput shape:")
print(model.input_shape)

print("\nOutput shape:")
print(model.output_shape)


# ============================================================
# COLLECT IMAGES
# ============================================================

images = []
true_labels = []
file_names = []

print("\n" + "=" * 70)
print("SCANNING TEST DATASET")
print("=" * 70)

for class_index, class_name in enumerate(CLASS_NAMES):

    class_dir = os.path.join(TEST_DIR, class_name)

    if not os.path.exists(class_dir):
        print(f"\nWARNING: Missing folder: {class_dir}")
        continue

    class_files = []

    for filename in os.listdir(class_dir):

        if filename.lower().endswith(VALID_EXTENSIONS):

            class_files.append(
                os.path.join(class_dir, filename)
            )

    print(f"{class_name:15s}: {len(class_files)} images")

    for file_path in class_files:

        try:

            img = image.load_img(
                file_path,
                target_size=IMG_SIZE
            )

            img_array = image.img_to_array(img)

            # IMPORTANT:
            # Keep pixel values in 0-255.
            # EfficientNet performs its own input rescaling.

            img_array = img_array.astype("float32")

            images.append(img_array)
            true_labels.append(class_index)
            file_names.append(file_path)

        except Exception as e:

            print("Skipping:", file_path)
            print("Reason:", e)


# ============================================================
# CONVERT TO NUMPY
# ============================================================

X = np.array(images, dtype=np.float32)
y_true = np.array(true_labels)

print("\nTotal images:", len(X))

if len(X) == 0:
    raise ValueError("No test images found.")


print("Input shape:", X.shape)
print("Input min:", X.min())
print("Input max:", X.max())


# ============================================================
# PREDICTION
# ============================================================

print("\n" + "=" * 70)
print("RUNNING PREDICTIONS")
print("=" * 70)

predictions = model.predict(
    X,
    batch_size=32,
    verbose=1
)

y_pred = np.argmax(predictions, axis=1)


# ============================================================
# ACCURACY
# ============================================================

accuracy = accuracy_score(
    y_true,
    y_pred
)

print("\n" + "=" * 70)
print("RESULTS")
print("=" * 70)

print(f"\nAccuracy: {accuracy * 100:.2f}%")


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
        labels=list(range(len(CLASS_NAMES))),
        target_names=CLASS_NAMES,
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
    labels=list(range(len(CLASS_NAMES)))
)

print("\nRows = Actual")
print("Columns = Predicted\n")

print(
    f"{'':18s}" +
    "".join(f"{c[:12]:>14s}" for c in CLASS_NAMES)
)

for i, class_name in enumerate(CLASS_NAMES):

    row = f"{class_name:18s}"

    for value in cm[i]:

        row += f"{value:14d}"

    print(row)


# ============================================================
# PREDICTION DISTRIBUTION
# ============================================================

print("\n" + "=" * 70)
print("PREDICTION DISTRIBUTION")
print("=" * 70)

for i, class_name in enumerate(CLASS_NAMES):

    count = np.sum(y_pred == i)

    percentage = (
        count / len(y_pred) * 100
    )

    print(
        f"{class_name:15s}: "
        f"{count:5d} "
        f"({percentage:6.2f}%)"
    )


# ============================================================
# SAVE RESULTS
# ============================================================

results_dir = r"D:\TruthLens\Results"

os.makedirs(
    results_dir,
    exist_ok=True
)

result_file = os.path.join(
    results_dir,
    "keras_internal_evaluation.txt"
)

with open(result_file, "w") as f:

    f.write(
        "TRUTHLENS INTERNAL KERAS EVALUATION\n"
    )

    f.write("=" * 60 + "\n")

    f.write(
        f"Model: {MODEL_PATH}\n"
    )

    f.write(
        f"Dataset: {TEST_DIR}\n\n"
    )

    f.write(
        f"Accuracy: {accuracy * 100:.2f}%\n\n"
    )

    f.write(
        classification_report(
            y_true,
            y_pred,
            labels=list(range(len(CLASS_NAMES))),
            target_names=CLASS_NAMES,
            zero_division=0
        )
    )

    f.write("\n\nCONFUSION MATRIX\n")
    f.write(str(cm))

print("\nResults saved to:")
print(result_file)

print("\n" + "=" * 70)
print("EVALUATION COMPLETE")
print("=" * 70)