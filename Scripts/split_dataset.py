import os
import shutil
import random
from pathlib import Path
from tqdm import tqdm


# ============================================================
# CONFIGURATION
# ============================================================

# Source folder containing processed images
SOURCE_DIR = Path(r"D:\TruthLens\Processed_Dataset")

# Destination folder where Train, Validation and Test will be created
OUTPUT_DIR = Path(r"D:\TruthLens\Split_Dataset")

# Dataset split ratios
TRAIN_RATIO = 0.80
VALIDATION_RATIO = 0.10
TEST_RATIO = 0.10

# Random seed ensures the same split every time
RANDOM_SEED = 42

# Supported image extensions
VALID_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp"
}


# ============================================================
# CHECK SPLIT RATIOS
# ============================================================

if abs((TRAIN_RATIO + VALIDATION_RATIO + TEST_RATIO) - 1.0) > 1e-9:
    raise ValueError(
        "Train, Validation and Test ratios must add up to 1.0"
    )


# ============================================================
# GET CLASS FOLDERS
# ============================================================

def get_class_folders():
    """
    Automatically detects all class folders inside Processed_Dataset.
    """

    if not SOURCE_DIR.exists():
        raise FileNotFoundError(
            f"Source folder not found:\n{SOURCE_DIR}"
        )

    class_folders = [
        folder
        for folder in SOURCE_DIR.iterdir()
        if folder.is_dir()
    ]

    if not class_folders:
        raise ValueError(
            f"No class folders found inside:\n{SOURCE_DIR}"
        )

    return sorted(class_folders)


# ============================================================
# GET VALID IMAGES
# ============================================================

def get_images(class_folder):
    """
    Returns all supported image files from a class folder.
    """

    return [
        file
        for file in class_folder.rglob("*")
        if file.is_file()
        and file.suffix.lower() in VALID_EXTENSIONS
    ]


# ============================================================
# CREATE OUTPUT FOLDERS
# ============================================================

def create_output_folders(class_names):
    """
    Creates Train, Validation and Test folders
    for every class.
    """

    for split_name in ["Train", "Validation", "Test"]:

        for class_name in class_names:

            folder_path = OUTPUT_DIR / split_name / class_name

            folder_path.mkdir(
                parents=True,
                exist_ok=True
            )


# ============================================================
# COPY FILE SAFELY
# ============================================================

def copy_file_safely(source_file, destination_folder):
    """
    Copies an image without overwriting another image
    having the same filename.
    """

    destination_file = destination_folder / source_file.name

    counter = 1

    while destination_file.exists():

        new_name = (
            f"{source_file.stem}_{counter}"
            f"{source_file.suffix}"
        )

        destination_file = destination_folder / new_name

        counter += 1

    shutil.copy2(source_file, destination_file)


# ============================================================
# SPLIT DATASET
# ============================================================

def split_dataset():

    print("\n" + "=" * 65)
    print("TRUTHLENS DATASET SPLITTING")
    print("=" * 65)

    print(f"\nSource folder:")
    print(SOURCE_DIR)

    print(f"\nOutput folder:")
    print(OUTPUT_DIR)

    print("\nSplit ratios:")
    print(f"Training   : {TRAIN_RATIO * 100:.0f}%")
    print(f"Validation : {VALIDATION_RATIO * 100:.0f}%")
    print(f"Testing    : {TEST_RATIO * 100:.0f}%")

    # Set random seed
    random.seed(RANDOM_SEED)

    # Find class folders
    class_folders = get_class_folders()

    class_names = [
        folder.name
        for folder in class_folders
    ]

    print("\nClasses detected:")

    for class_name in class_names:
        print(f"  - {class_name}")

    # Create output directories
    create_output_folders(class_names)

    # Store final statistics
    statistics = {}

    # Process every class
    for class_folder in class_folders:

        class_name = class_folder.name

        print("\n" + "-" * 65)
        print(f"Processing class: {class_name}")
        print("-" * 65)

        # Get all images
        images = get_images(class_folder)

        total_images = len(images)

        if total_images == 0:

            print(
                f"Warning: No valid images found "
                f"in class '{class_name}'"
            )

            statistics[class_name] = {
                "Total": 0,
                "Train": 0,
                "Validation": 0,
                "Test": 0
            }

            continue

        print(f"Total images found: {total_images}")

        # Shuffle images randomly
        random.shuffle(images)

        # Calculate split sizes
        train_count = int(total_images * TRAIN_RATIO)

        validation_count = int(
            total_images * VALIDATION_RATIO
        )

        # Remaining images go to test set
        test_count = (
            total_images
            - train_count
            - validation_count
        )

        # Create split lists
        train_images = images[:train_count]

        validation_images = images[
            train_count:
            train_count + validation_count
        ]

        test_images = images[
            train_count + validation_count:
        ]

        split_data = {
            "Train": train_images,
            "Validation": validation_images,
            "Test": test_images
        }

        # Copy images
        for split_name, split_images in split_data.items():

            destination_folder = (
                OUTPUT_DIR
                / split_name
                / class_name
            )

            print(
                f"\nCopying {split_name}: "
                f"{len(split_images)} images"
            )

            for image_path in tqdm(
                split_images,
                desc=f"{class_name} -> {split_name}",
                unit="image"
            ):

                try:

                    copy_file_safely(
                        image_path,
                        destination_folder
                    )

                except Exception as error:

                    print(
                        f"\nCould not copy:\n"
                        f"{image_path}\n"
                        f"Reason: {error}"
                    )

        # Save statistics
        statistics[class_name] = {
            "Total": total_images,
            "Train": len(train_images),
            "Validation": len(validation_images),
            "Test": len(test_images)
        }


    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print("\n\n" + "=" * 65)
    print("DATASET SPLITTING COMPLETED")
    print("=" * 65)

    print(
        f"\n{'Class':<20}"
        f"{'Total':>10}"
        f"{'Train':>10}"
        f"{'Validation':>15}"
        f"{'Test':>10}"
    )

    print("-" * 65)

    grand_total = 0
    grand_train = 0
    grand_validation = 0
    grand_test = 0

    for class_name, values in statistics.items():

        print(
            f"{class_name:<20}"
            f"{values['Total']:>10}"
            f"{values['Train']:>10}"
            f"{values['Validation']:>15}"
            f"{values['Test']:>10}"
        )

        grand_total += values["Total"]
        grand_train += values["Train"]
        grand_validation += values["Validation"]
        grand_test += values["Test"]

    print("-" * 65)

    print(
        f"{'TOTAL':<20}"
        f"{grand_total:>10}"
        f"{grand_train:>10}"
        f"{grand_validation:>15}"
        f"{grand_test:>10}"
    )

    print("\nYour original Processed_Dataset was NOT modified.")

    print(
        "\nSplit dataset saved successfully at:"
    )

    print(OUTPUT_DIR)

    print("\nNext step: Dataset balance analysis and model training.")


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    split_dataset()