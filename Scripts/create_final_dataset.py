from pathlib import Path
import random
import shutil
from PIL import Image
from tqdm import tqdm


# ============================================================
# TRUTHLENS - FINAL BALANCED DATASET CREATOR
# ============================================================

# Existing TruthLens split dataset
TRUTHLENS_SPLIT_DIR = Path(r"D:\TruthLens\Split_Dataset")

# Downloaded Deepfake dataset
DEEPFAKE_DATASET_DIR = Path(
    r"D:\TruthLens\Downloaded_Datasets\Deepfake_Real_Images\Dataset"
)

# Final dataset destination
FINAL_DATASET_DIR = Path(r"D:\TruthLens\Final_Dataset")


# ============================================================
# CONFIGURATION
# ============================================================

CLASSES = [
    "AI_Generated",
    "Deepfake",
    "Manipulated",
    "Real"
]

SPLITS = [
    "Train",
    "Validation",
    "Test"
]

TARGET_COUNTS = {
    "Train": 11200,
    "Validation": 1400,
    "Test": 1400
}

VALID_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp"
}

RANDOM_SEED = 42

# All final images will be standardized to this size
IMAGE_SIZE = (224, 224)


# ============================================================
# GET IMAGES
# ============================================================

def get_images(folder):
    """
    Returns all supported images inside a folder.
    """

    if not folder.exists():
        return []

    return sorted([
        file
        for file in folder.rglob("*")
        if file.is_file()
        and file.suffix.lower() in VALID_EXTENSIONS
    ])


# ============================================================
# GET SOURCE FOLDER
# ============================================================

def get_source_folder(split_name, class_name):
    """
    Determines where images for each class should come from.

    Deepfake:
        Uses the newly downloaded dataset.

    Other classes:
        Uses the existing TruthLens Split_Dataset.
    """

    if class_name == "Deepfake":

        return (
            DEEPFAKE_DATASET_DIR
            / split_name
            / "Fake"
        )

    return (
        TRUTHLENS_SPLIT_DIR
        / split_name
        / class_name
    )


# ============================================================
# CREATE OUTPUT FOLDERS
# ============================================================

def create_output_folders():

    for split_name in SPLITS:

        for class_name in CLASSES:

            folder = (
                FINAL_DATASET_DIR
                / split_name
                / class_name
            )

            folder.mkdir(
                parents=True,
                exist_ok=True
            )


# ============================================================
# SAVE IMAGE
# ============================================================

def process_and_save_image(
    source_path,
    destination_path
):
    """
    Opens an image, converts it to RGB,
    resizes to 224x224 and saves as JPEG.
    """

    with Image.open(source_path) as image:

        image = image.convert("RGB")

        image = image.resize(
            IMAGE_SIZE,
            Image.Resampling.LANCZOS
        )

        image.save(
            destination_path,
            "JPEG",
            quality=95
        )


# ============================================================
# CHECK AVAILABLE DATA
# ============================================================

def check_sources():

    print("\nChecking available source images...\n")

    everything_ok = True

    for split_name in SPLITS:

        required = TARGET_COUNTS[split_name]

        print(split_name.upper())
        print("-" * 60)

        for class_name in CLASSES:

            source_folder = get_source_folder(
                split_name,
                class_name
            )

            images = get_images(source_folder)

            available = len(images)

            status = "OK"

            if available < required:
                status = "NOT ENOUGH"
                everything_ok = False

            print(
                f"{class_name:<15}"
                f"Available: {available:>8,}   "
                f"Required: {required:>6,}   "
                f"{status}"
            )

        print()

    return everything_ok


# ============================================================
# BUILD FINAL DATASET
# ============================================================

def build_final_dataset():

    print("=" * 70)
    print("TRUTHLENS - FINAL BALANCED DATASET CREATOR")
    print("=" * 70)

    print(f"\nOutput:")
    print(FINAL_DATASET_DIR)

    print("\nTarget per class:")

    for split_name in SPLITS:

        print(
            f"{split_name:<12}: "
            f"{TARGET_COUNTS[split_name]:,}"
        )

    print("\nTotal per class: 14,000")
    print("Final dataset total: 56,000")

    # --------------------------------------------------------
    # SAFETY CHECK
    # --------------------------------------------------------

    if FINAL_DATASET_DIR.exists():

        existing_images = get_images(
            FINAL_DATASET_DIR
        )

        if existing_images:

            print("\nERROR:")
            print(
                "Final_Dataset already contains images."
            )

            print(
                "The script stopped to prevent duplicate files."
            )

            print(
                "\nIf you intentionally want to rebuild it,"
                " rename or delete Final_Dataset first."
            )

            return

    # --------------------------------------------------------
    # CHECK SOURCES
    # --------------------------------------------------------

    if not check_sources():

        print("=" * 70)
        print("DATASET CREATION CANCELLED")
        print("=" * 70)

        print(
            "\nAt least one source does not contain "
            "enough images."
        )

        print(
            "No final dataset was created."
        )

        return

    # Create directories only after successful source check
    create_output_folders()

    random.seed(RANDOM_SEED)

    statistics = {}

    # --------------------------------------------------------
    # PROCESS SPLITS
    # --------------------------------------------------------

    for split_name in SPLITS:

        statistics[split_name] = {}

        target_count = TARGET_COUNTS[split_name]

        print("\n" + "=" * 70)
        print(f"CREATING {split_name.upper()} SET")
        print("=" * 70)

        for class_name in CLASSES:

            source_folder = get_source_folder(
                split_name,
                class_name
            )

            destination_folder = (
                FINAL_DATASET_DIR
                / split_name
                / class_name
            )

            images = get_images(source_folder)

            # Use independent deterministic sampling
            # for each class/split.
            rng = random.Random(
                f"{RANDOM_SEED}-{split_name}-{class_name}"
            )

            selected_images = rng.sample(
                images,
                target_count
            )

            successful = 0
            failed = 0

            print(
                f"\n{class_name}"
                f" ({target_count:,} images)"
            )

            for index, image_path in enumerate(
                tqdm(
                    selected_images,
                    desc=f"{split_name}/{class_name}",
                    unit="img"
                ),
                start=1
            ):

                # Prefix makes every output filename unique
                output_name = (
                    f"{class_name.lower()}_"
                    f"{index:06d}.jpg"
                )

                destination_path = (
                    destination_folder
                    / output_name
                )

                try:

                    process_and_save_image(
                        image_path,
                        destination_path
                    )

                    successful += 1

                except Exception as error:

                    failed += 1

                    print(
                        f"\nFailed: {image_path}"
                    )

                    print(
                        f"Reason: {error}"
                    )

            statistics[split_name][class_name] = {
                "successful": successful,
                "failed": failed
            }

    # --------------------------------------------------------
    # FINAL SUMMARY
    # --------------------------------------------------------

    print("\n\n" + "=" * 70)
    print("FINAL DATASET SUMMARY")
    print("=" * 70)

    grand_total = 0

    for split_name in SPLITS:

        print(f"\n{split_name.upper()}")
        print("-" * 45)

        split_total = 0

        for class_name in CLASSES:

            values = statistics[
                split_name
            ][class_name]

            successful = values["successful"]
            failed = values["failed"]

            split_total += successful

            print(
                f"{class_name:<15}"
                f"{successful:>8,}"
                f"   Failed: {failed}"
            )

        print("-" * 45)

        print(
            f"{'TOTAL':<15}"
            f"{split_total:>8,}"
        )

        grand_total += split_total

    print("\n" + "=" * 70)

    print(
        f"TOTAL FINAL IMAGES: {grand_total:,}"
    )

    print("=" * 70)

    if grand_total == 56000:

        print(
            "\nSUCCESS: Balanced dataset created."
        )

    else:

        print(
            "\nWARNING: Final image count is not 56,000."
        )

        print(
            "Check failed image counts above."
        )

    print(
        "\nFinal dataset location:"
    )

    print(FINAL_DATASET_DIR)


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    build_final_dataset()