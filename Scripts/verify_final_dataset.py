from pathlib import Path
from PIL import Image
import hashlib
from collections import defaultdict
from tqdm import tqdm


# ============================================================
# TRUTHLENS - FINAL DATASET VERIFICATION
# ============================================================

DATASET_DIR = Path(r"D:\TruthLens\Final_Dataset")

SPLITS = ["Train", "Validation", "Test"]

CLASSES = [
    "AI_Generated",
    "Deepfake",
    "Manipulated",
    "Real"
]

EXPECTED_COUNTS = {
    "Train": 11200,
    "Validation": 1400,
    "Test": 1400
}

EXPECTED_SIZE = (224, 224)
EXPECTED_MODE = "RGB"

VALID_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp"
}


# ============================================================
# GET IMAGE FILES
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
# CALCULATE FILE HASH
# ============================================================

def calculate_hash(file_path):
    """
    SHA-256 hash used to identify exact duplicate files.
    """

    sha256 = hashlib.sha256()

    with open(file_path, "rb") as file:

        while True:

            chunk = file.read(1024 * 1024)

            if not chunk:
                break

            sha256.update(chunk)

    return sha256.hexdigest()


# ============================================================
# MAIN VERIFICATION
# ============================================================

def verify_dataset():

    print("=" * 70)
    print("TRUTHLENS - FINAL DATASET VERIFICATION")
    print("=" * 70)

    print(f"\nDataset:")
    print(DATASET_DIR)

    if not DATASET_DIR.exists():

        print("\nERROR: Final_Dataset does not exist.")
        return

    # --------------------------------------------------------
    # STATISTICS
    # --------------------------------------------------------

    total_images = 0

    corrupted_images = []
    wrong_size_images = []
    wrong_mode_images = []

    # Hash -> list of (split, class, path)
    hash_locations = defaultdict(list)

    counts = {}

    # --------------------------------------------------------
    # VERIFY EACH SPLIT
    # --------------------------------------------------------

    for split_name in SPLITS:

        counts[split_name] = {}

        print("\n" + "=" * 70)
        print(split_name.upper())
        print("=" * 70)

        for class_name in CLASSES:

            folder = (
                DATASET_DIR
                / split_name
                / class_name
            )

            print(f"\nChecking {split_name}/{class_name}")

            if not folder.exists():

                print("ERROR: Folder does not exist.")

                counts[split_name][class_name] = 0
                continue

            images = get_images(folder)

            image_count = len(images)

            counts[split_name][class_name] = image_count

            total_images += image_count

            print(
                f"Images found: {image_count:,}"
            )

            for image_path in tqdm(
                images,
                desc=f"{split_name}/{class_name}",
                unit="img"
            ):

                # --------------------------------------------
                # IMAGE INTEGRITY / SIZE / MODE
                # --------------------------------------------

                try:

                    with Image.open(image_path) as image:

                        # Force Pillow to decode image data
                        image.load()

                        if image.size != EXPECTED_SIZE:

                            wrong_size_images.append(
                                (
                                    image_path,
                                    image.size
                                )
                            )

                        if image.mode != EXPECTED_MODE:

                            wrong_mode_images.append(
                                (
                                    image_path,
                                    image.mode
                                )
                            )

                except Exception as error:

                    corrupted_images.append(
                        (
                            image_path,
                            str(error)
                        )
                    )

                    # Don't hash corrupted files
                    continue

                # --------------------------------------------
                # HASH
                # --------------------------------------------

                try:

                    file_hash = calculate_hash(
                        image_path
                    )

                    hash_locations[file_hash].append(
                        (
                            split_name,
                            class_name,
                            image_path
                        )
                    )

                except Exception as error:

                    print(
                        f"\nHash error: {image_path}"
                    )

                    print(error)

    # ========================================================
    # COUNT CHECK
    # ========================================================

    print("\n\n" + "=" * 70)
    print("IMAGE COUNT CHECK")
    print("=" * 70)

    count_errors = []

    for split_name in SPLITS:

        expected = EXPECTED_COUNTS[
            split_name
        ]

        print(f"\n{split_name.upper()}")

        for class_name in CLASSES:

            actual = counts[
                split_name
            ].get(
                class_name,
                0
            )

            status = "OK"

            if actual != expected:

                status = "ERROR"

                count_errors.append(
                    (
                        split_name,
                        class_name,
                        expected,
                        actual
                    )
                )

            print(
                f"{class_name:<15}"
                f"{actual:>8,}   "
                f"Expected: {expected:>8,}   "
                f"{status}"
            )

    # ========================================================
    # DUPLICATE CHECK
    # ========================================================

    print("\n\n" + "=" * 70)
    print("DUPLICATE CHECK")
    print("=" * 70)

    duplicate_groups = {
        file_hash: locations
        for file_hash, locations
        in hash_locations.items()
        if len(locations) > 1
    }

    print(
        f"\nExact duplicate groups: "
        f"{len(duplicate_groups):,}"
    )

    # --------------------------------------------------------
    # CROSS-SPLIT LEAKAGE
    # --------------------------------------------------------

    cross_split_duplicates = {}

    same_split_duplicates = {}

    for file_hash, locations in duplicate_groups.items():

        split_set = {
            location[0]
            for location in locations
        }

        if len(split_set) > 1:

            cross_split_duplicates[
                file_hash
            ] = locations

        else:

            same_split_duplicates[
                file_hash
            ] = locations

    print(
        f"Duplicate groups inside same split: "
        f"{len(same_split_duplicates):,}"
    )

    print(
        f"Duplicate groups across splits: "
        f"{len(cross_split_duplicates):,}"
    )

    # ========================================================
    # SHOW CROSS-SPLIT DUPLICATES
    # ========================================================

    if cross_split_duplicates:

        print("\n" + "=" * 70)
        print("WARNING: CROSS-SPLIT DUPLICATES FOUND")
        print("=" * 70)

        # Show only first 20 groups to keep terminal readable
        for index, (
            file_hash,
            locations
        ) in enumerate(
            cross_split_duplicates.items(),
            start=1
        ):

            if index > 20:

                remaining = (
                    len(cross_split_duplicates)
                    - 20
                )

                print(
                    f"\n... and {remaining:,} "
                    f"more duplicate groups."
                )

                break

            print(
                f"\nDuplicate Group {index}"
            )

            print(
                f"Hash: {file_hash[:16]}..."
            )

            for (
                split_name,
                class_name,
                image_path
            ) in locations:

                print(
                    f"  {split_name:<12} "
                    f"{class_name:<15} "
                    f"{image_path.name}"
                )

    # ========================================================
    # FINAL REPORT
    # ========================================================

    print("\n\n" + "=" * 70)
    print("FINAL VERIFICATION REPORT")
    print("=" * 70)

    print(
        f"\nTotal images              : "
        f"{total_images:,}"
    )

    print(
        f"Count errors              : "
        f"{len(count_errors):,}"
    )

    print(
        f"Corrupted images          : "
        f"{len(corrupted_images):,}"
    )

    print(
        f"Wrong image dimensions    : "
        f"{len(wrong_size_images):,}"
    )

    print(
        f"Non-RGB images            : "
        f"{len(wrong_mode_images):,}"
    )

    print(
        f"Exact duplicate groups    : "
        f"{len(duplicate_groups):,}"
    )

    print(
        f"Cross-split duplicates    : "
        f"{len(cross_split_duplicates):,}"
    )

    # ========================================================
    # FINAL DECISION
    # ========================================================

    critical_errors = (
        len(count_errors)
        + len(corrupted_images)
        + len(wrong_size_images)
        + len(wrong_mode_images)
        + len(cross_split_duplicates)
    )

    print("\n" + "=" * 70)

    if critical_errors == 0:

        print("SUCCESS: DATASET PASSED ALL CRITICAL CHECKS")

        print(
            "\nThe Final_Dataset is ready "
            "for baseline model training."
        )

        if duplicate_groups:

            print(
                "\nNote: Some duplicates exist inside "
                "individual splits, but no duplicate "
                "content crosses Train/Validation/Test."
            )

    else:

        print("WARNING: DATASET NEEDS ATTENTION")

        print(
            "\nDo NOT start final model training yet."
        )

        print(
            "Review the issues reported above."
        )

    print("=" * 70)


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    verify_dataset()