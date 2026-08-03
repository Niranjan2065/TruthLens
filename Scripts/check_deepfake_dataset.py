from pathlib import Path
from PIL import Image

DATASET_DIR = Path(
    r"D:\TruthLens\Downloaded_Datasets\Deepfake_Real_Images\Dataset"
)

SPLITS = ["Train", "Validation", "Test"]
CLASSES = ["Fake", "Real"]

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def get_images(folder):
    return [
        file for file in folder.rglob("*")
        if file.is_file()
        and file.suffix.lower() in VALID_EXTENSIONS
    ]


def check_dataset():

    print("=" * 60)
    print("TRUTHLENS - DEEPFAKE DATASET CHECK")
    print("=" * 60)

    grand_total = 0

    for split in SPLITS:

        print(f"\n{split.upper()}")
        print("-" * 40)

        split_total = 0

        for class_name in CLASSES:

            folder = DATASET_DIR / split / class_name

            if not folder.exists():
                print(f"{class_name:<10}: FOLDER NOT FOUND")
                continue

            images = get_images(folder)

            valid = 0
            corrupted = 0

            print(f"\nChecking {split}/{class_name}...")

            for image_path in images:

                try:
                    with Image.open(image_path) as img:
                        img.verify()

                    valid += 1

                except Exception:
                    corrupted += 1

            print(f"{class_name:<10}: {len(images):,}")
            print(f"  Valid     : {valid:,}")
            print(f"  Corrupted : {corrupted:,}")

            split_total += valid

        print(f"\n{split} total: {split_total:,}")

        grand_total += split_total

    print("\n" + "=" * 60)
    print(f"TOTAL VALID IMAGES: {grand_total:,}")
    print("=" * 60)


if __name__ == "__main__":
    check_dataset()