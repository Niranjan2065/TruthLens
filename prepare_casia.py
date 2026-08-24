from pathlib import Path
from PIL import Image

# ============================================================
# TRUTHLENS - CASIA PREPROCESSING
# ============================================================

PROJECT_DIR = Path(r"D:\TruthLens")

SOURCE_DIR = PROJECT_DIR / "RealWorld_CASIA_Test"
OUTPUT_DIR = PROJECT_DIR / "CASIA_Processed"

IMAGE_SIZE = (224, 224)

CLASSES = [
    "Real",
    "Manipulated"
]

VALID_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp"
}

print("=" * 70)
print("TRUTHLENS - CASIA PREPROCESSING")
print("=" * 70)

total = 0

for class_name in CLASSES:

    source_folder = SOURCE_DIR / class_name
    output_folder = OUTPUT_DIR / class_name

    output_folder.mkdir(
        parents=True,
        exist_ok=True
    )

    images = [
        f for f in source_folder.iterdir()
        if f.is_file()
        and f.suffix.lower() in VALID_EXTENSIONS
    ]

    print(f"\n{class_name}: {len(images)} images")

    for index, image_path in enumerate(images, start=1):

        try:

            image = Image.open(image_path).convert("RGB")

            # Match TruthLens model input size
            image = image.resize(
                IMAGE_SIZE,
                Image.Resampling.LANCZOS
            )

            output_path = (
                output_folder
                / f"{class_name.lower()}_{index:03d}.jpg"
            )

            image.save(
                output_path,
                "JPEG",
                quality=95
            )

            total += 1

        except Exception as e:

            print(
                f"ERROR: {image_path.name} -> {e}"
            )

    print(
        f"Processed {class_name}: "
        f"{len(list(output_folder.glob('*.jpg')))}"
    )

print("\n" + "=" * 70)
print("PREPROCESSING COMPLETE")
print("=" * 70)

print(f"\nTotal images processed: {total}")

print("\nOutput:")
print(OUTPUT_DIR)

print("\nExpected structure:")
print(r"D:\TruthLens\CASIA_Processed\Real")
print(r"D:\TruthLens\CASIA_Processed\Manipulated")