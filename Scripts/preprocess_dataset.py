"""
TruthLens Dataset Preprocessing
--------------------------------
✓ Searches all subfolders automatically
✓ Skips corrupted images
✓ Converts images to RGB
✓ Resizes to 224x224
✓ Saves to Processed_Dataset
"""

import os
from PIL import Image

# ------------------------------
# Paths
# ------------------------------

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

INPUT_FOLDER = os.path.join(BASE_DIR, "Dataset")
OUTPUT_FOLDER = os.path.join(BASE_DIR, "Processed_Dataset")

CLASSES = [
    "Real",
    "AI_Generated",
    "Manipulated",
    "Deepfake"
]

IMAGE_SIZE = (224, 224)

SUPPORTED_EXTENSIONS = (
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
    ".tif",
    ".tiff"
)

# ------------------------------
# Statistics
# ------------------------------

total = 0
processed = 0
skipped = 0

print("=" * 60)
print("TruthLens Dataset Preprocessing")
print("=" * 60)

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

for class_name in CLASSES:

    input_dir = os.path.join(INPUT_FOLDER, class_name)
    output_dir = os.path.join(OUTPUT_FOLDER, class_name)

    os.makedirs(output_dir, exist_ok=True)

    if not os.path.exists(input_dir):
        print(f"\nFolder not found: {input_dir}")
        continue

    # Find images recursively
    image_paths = []

    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file.lower().endswith(SUPPORTED_EXTENSIONS):
                image_paths.append(os.path.join(root, file))

    print(f"\nProcessing: {class_name}")
    print("-" * 50)
    print(f"Images Found: {len(image_paths)}")

    class_processed = 0
    class_skipped = 0

    for i, image_path in enumerate(image_paths, start=1):

        total += 1

        filename = os.path.basename(image_path)
        output_path = os.path.join(output_dir, filename)

        try:
            with Image.open(image_path) as img:

                img = img.convert("RGB")
                img = img.resize(
                    IMAGE_SIZE,
                    Image.Resampling.LANCZOS
                )

                img.save(output_path)

                processed += 1
                class_processed += 1

                print(
                    f"\rProcessed {i}/{len(image_paths)}",
                    end=""
                )

        except Exception as e:

            skipped += 1
            class_skipped += 1

            print(f"\nSkipped: {image_path}")

    print("\nCompleted")
    print(f"Processed : {class_processed}")
    print(f"Skipped   : {class_skipped}")

print("\n" + "=" * 60)
print("PREPROCESSING FINISHED")
print("=" * 60)

print(f"Total Images     : {total}")
print(f"Processed Images : {processed}")
print(f"Skipped Images   : {skipped}")

print("\nOutput Folder:")
print(OUTPUT_FOLDER)