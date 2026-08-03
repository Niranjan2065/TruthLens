from pathlib import Path
from PIL import Image
from tqdm import tqdm
import hashlib
import random
import shutil


# ============================================================
# TRUTHLENS - FINAL DATASET DUPLICATE CLEANER
# ============================================================

FINAL_DATASET_DIR = Path(r"D:\TruthLens\Final_Dataset")

# Original processed/split dataset
ORIGINAL_SPLIT_DIR = Path(r"D:\TruthLens\Split_Dataset")

# Downloaded Deepfake dataset
DEEPFAKE_DATASET_DIR = Path(
    r"D:\TruthLens\Downloaded_Datasets\Deepfake_Real_Images\Dataset"
)

SPLITS = [
    "Train",
    "Validation",
    "Test"
]

CLASSES = [
    "AI_Generated",
    "Deepfake",
    "Manipulated",
    "Real"
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

IMAGE_SIZE = (224, 224)

RANDOM_SEED = 42


# ============================================================
# GET IMAGES
# ============================================================

def get_images(folder):

    if not folder.exists():
        return []

    return sorted([
        file
        for file in folder.rglob("*")
        if file.is_file()
        and file.suffix.lower() in VALID_EXTENSIONS
    ])


# ============================================================
# HASH FILE
# ============================================================

def calculate_hash(file_path):

    sha256 = hashlib.sha256()

    with open(file_path, "rb") as file:

        while True:

            chunk = file.read(1024 * 1024)

            if not chunk:
                break

            sha256.update(chunk)

    return sha256.hexdigest()


# ============================================================
# PROCESS SOURCE IMAGE
# ============================================================

def process_image(source_path, destination_path):
    """
    Converts source image to the exact same final format used
    when Final_Dataset was created.

    This is important because duplicate checking should happen
    on the final standardized representation.
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
# GET SOURCE DIRECTORY
# ============================================================

def get_source_directory(split_name, class_name):

    if class_name == "Deepfake":

        return (
            DEEPFAKE_DATASET_DIR
            / split_name
            / "Fake"
        )

    return (
        ORIGINAL_SPLIT_DIR
        / split_name
        / class_name
    )


# ============================================================
# TEMP DIRECTORY
# ============================================================

TEMP_DIR = Path(
    r"D:\TruthLens\Temp_Duplicate_Check"
)


# ============================================================
# FIND DUPLICATES IN FINAL DATASET
# ============================================================

def scan_final_dataset():

    print("\n" + "=" * 70)
    print("STEP 1 - SCANNING FINAL DATASET")
    print("=" * 70)

    # Track hashes separately per class.
    #
    # If identical content somehow appears in two different
    # semantic classes, that is a label-conflict problem and
    # should not be silently cleaned here.
    class_hashes = {
        class_name: {}
        for class_name in CLASSES
    }

    duplicates_to_remove = []

    duplicate_count = 0

    for class_name in CLASSES:

        print(
            f"\nScanning class: {class_name}"
        )

        for split_name in SPLITS:

            folder = (
                FINAL_DATASET_DIR
                / split_name
                / class_name
            )

            images = get_images(folder)

            for image_path in tqdm(
                images,
                desc=f"{split_name}/{class_name}",
                unit="img"
            ):

                try:

                    image_hash = calculate_hash(
                        image_path
                    )

                except Exception as error:

                    print(
                        f"\nCould not hash: {image_path}"
                    )

                    print(error)

                    continue

                if image_hash not in class_hashes[class_name]:

                    # Keep first occurrence.
                    #
                    # Because SPLITS begins with Train,
                    # duplicates preferentially remain in
                    # Train rather than Validation/Test.
                    class_hashes[class_name][image_hash] = (
                        split_name,
                        image_path
                    )

                else:

                    duplicate_count += 1

                    original_split, original_path = (
                        class_hashes[class_name][image_hash]
                    )

                    duplicates_to_remove.append(
                        (
                            split_name,
                            class_name,
                            image_path,
                            original_split,
                            original_path
                        )
                    )

    print(
        f"\nDuplicate files to replace: "
        f"{duplicate_count:,}"
    )

    return duplicates_to_remove


# ============================================================
# REMOVE DUPLICATES
# ============================================================

def remove_duplicates(duplicates):

    print("\n" + "=" * 70)
    print("STEP 2 - REMOVING DUPLICATES")
    print("=" * 70)

    removed_by_split_class = {}

    for split_name in SPLITS:

        removed_by_split_class[split_name] = {}

        for class_name in CLASSES:

            removed_by_split_class[
                split_name
            ][class_name] = 0

    for (
        split_name,
        class_name,
        duplicate_path,
        original_split,
        original_path
    ) in duplicates:

        if duplicate_path.exists():

            try:

                duplicate_path.unlink()

                removed_by_split_class[
                    split_name
                ][class_name] += 1

            except Exception as error:

                print(
                    f"\nCould not delete:"
                    f"\n{duplicate_path}"
                )

                print(error)

    print("\nRemoved:")

    total_removed = 0

    for split_name in SPLITS:

        print(f"\n{split_name}")

        for class_name in CLASSES:

            removed = (
                removed_by_split_class[
                    split_name
                ][class_name]
            )

            print(
                f"  {class_name:<15}: "
                f"{removed:,}"
            )

            total_removed += removed

    print(
        f"\nTotal removed: {total_removed:,}"
    )

    return removed_by_split_class


# ============================================================
# BUILD EXISTING HASH SET
# ============================================================

def build_existing_hashes(class_name):

    hashes = set()

    for split_name in SPLITS:

        folder = (
            FINAL_DATASET_DIR
            / split_name
            / class_name
        )

        for image_path in get_images(folder):

            try:

                hashes.add(
                    calculate_hash(image_path)
                )

            except Exception:
                pass

    return hashes


# ============================================================
# GET NEXT OUTPUT NUMBER
# ============================================================

def get_next_number(folder, class_name):

    maximum = 0

    prefix = f"{class_name.lower()}_"

    for file_path in get_images(folder):

        stem = file_path.stem

        if not stem.startswith(prefix):
            continue

        number_text = stem[len(prefix):]

        try:

            number = int(number_text)

            maximum = max(
                maximum,
                number
            )

        except ValueError:
            pass

    return maximum + 1


# ============================================================
# REFILL DATASET
# ============================================================

def refill_dataset():

    print("\n" + "=" * 70)
    print("STEP 3 - REFILLING DATASET")
    print("=" * 70)

    TEMP_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    random.seed(RANDOM_SEED)

    total_added = 0

    for class_name in CLASSES:

        print(
            "\n" + "-" * 70
        )

        print(
            f"CLASS: {class_name}"
        )

        print(
            "-" * 70
        )

        # Hashes currently present across ALL splits
        # of this class.
        existing_hashes = build_existing_hashes(
            class_name
        )

        for split_name in SPLITS:

            destination_folder = (
                FINAL_DATASET_DIR
                / split_name
                / class_name
            )

            current_images = get_images(
                destination_folder
            )

            current_count = len(
                current_images
            )

            target_count = TARGET_COUNTS[
                split_name
            ]

            needed = (
                target_count
                - current_count
            )

            print(
                f"\n{split_name}/{class_name}"
            )

            print(
                f"Current : {current_count:,}"
            )

            print(
                f"Target  : {target_count:,}"
            )

            print(
                f"Need    : {needed:,}"
            )

            if needed <= 0:

                print("No refill needed.")

                continue

            source_folder = get_source_directory(
                split_name,
                class_name
            )

            candidates = get_images(
                source_folder
            )

            if not candidates:

                raise RuntimeError(
                    f"No source images found:\n"
                    f"{source_folder}"
                )

            rng = random.Random(
                f"{RANDOM_SEED}-"
                f"{split_name}-"
                f"{class_name}-refill"
            )

            rng.shuffle(candidates)

            next_number = get_next_number(
                destination_folder,
                class_name
            )

            added = 0

            checked = 0

            for source_path in tqdm(
                candidates,
                desc="Searching replacements",
                unit="img"
            ):

                if added >= needed:
                    break

                checked += 1

                # --------------------------------------------
                # Standardize candidate temporarily
                # --------------------------------------------

                temp_path = (
                    TEMP_DIR
                    / "candidate.jpg"
                )

                try:

                    process_image(
                        source_path,
                        temp_path
                    )

                    candidate_hash = (
                        calculate_hash(
                            temp_path
                        )
                    )

                except Exception:

                    if temp_path.exists():
                        temp_path.unlink()

                    continue

                # --------------------------------------------
                # Reject duplicate
                # --------------------------------------------

                if candidate_hash in existing_hashes:

                    if temp_path.exists():
                        temp_path.unlink()

                    continue

                # --------------------------------------------
                # Candidate is unique
                # --------------------------------------------

                output_name = (
                    f"{class_name.lower()}_"
                    f"{next_number:06d}.jpg"
                )

                destination_path = (
                    destination_folder
                    / output_name
                )

                shutil.move(
                    str(temp_path),
                    str(destination_path)
                )

                existing_hashes.add(
                    candidate_hash
                )

                next_number += 1
                added += 1
                total_added += 1

            print(
                f"\nAdded   : {added:,}"
            )

            print(
                f"Checked : {checked:,}"
            )

            if added < needed:

                raise RuntimeError(
                    f"\nCould not find enough unique "
                    f"replacement images for "
                    f"{split_name}/{class_name}.\n"
                    f"Needed: {needed}\n"
                    f"Added: {added}"
                )

    # Clean temp directory

    if TEMP_DIR.exists():

        try:

            shutil.rmtree(
                TEMP_DIR
            )

        except Exception:
            pass

    print(
        f"\nTotal replacements added: "
        f"{total_added:,}"
    )


# ============================================================
# FINAL COUNT CHECK
# ============================================================

def final_count_check():

    print("\n" + "=" * 70)
    print("STEP 4 - FINAL COUNT CHECK")
    print("=" * 70)

    everything_ok = True

    grand_total = 0

    for split_name in SPLITS:

        print(
            f"\n{split_name.upper()}"
        )

        split_total = 0

        for class_name in CLASSES:

            folder = (
                FINAL_DATASET_DIR
                / split_name
                / class_name
            )

            count = len(
                get_images(folder)
            )

            expected = TARGET_COUNTS[
                split_name
            ]

            status = "OK"

            if count != expected:

                status = "ERROR"

                everything_ok = False

            print(
                f"{class_name:<15}"
                f"{count:>8,}   "
                f"Expected: {expected:>8,}   "
                f"{status}"
            )

            split_total += count

        print(
            f"TOTAL{'':<10}"
            f"{split_total:>8,}"
        )

        grand_total += split_total

    print("\n" + "=" * 70)

    print(
        f"TOTAL IMAGES: {grand_total:,}"
    )

    print("=" * 70)

    if (
        everything_ok
        and grand_total == 56000
    ):

        print(
            "\nSUCCESS: Dataset counts restored."
        )

    else:

        print(
            "\nWARNING: Dataset counts "
            "do not match expected values."
        )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("TRUTHLENS - FINAL DATASET CLEANER")
    print("=" * 70)

    print(
        "\nThis script will:"
    )

    print(
        "1. Find exact duplicate images."
    )

    print(
        "2. Keep one copy of each image per class."
    )

    print(
        "3. Remove additional copies."
    )

    print(
        "4. Replace them with unused unique images."
    )

    print(
        "5. Restore the 56,000-image balance."
    )

    duplicates = scan_final_dataset()

    if not duplicates:

        print(
            "\nNo exact duplicates found."
        )

        final_count_check()

        return

    remove_duplicates(
        duplicates
    )

    refill_dataset()

    final_count_check()

    print("\n" + "=" * 70)

    print(
        "CLEANING COMPLETED"
    )

    print("=" * 70)

    print(
        "\nIMPORTANT:"
    )

    print(
        "Run verify_final_dataset.py again "
        "before model training."
    )


if __name__ == "__main__":

    main()