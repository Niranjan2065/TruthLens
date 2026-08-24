from pathlib import Path
import shutil

# ============================================================
# TRUTHLENS - PREPARE IMD2020 EXTERNAL TEST DATASET
# ============================================================

SOURCE = Path(r"D:\IMD2020")

OUTPUT = Path(r"D:\TruthLens\External_Test_IMD2020")

REAL_DIR = OUTPUT / "Real"
MANIPULATED_DIR = OUTPUT / "Manipulated"

TARGET_PER_CLASS = 100

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff"
}

# ------------------------------------------------------------
# Create output folders
# ------------------------------------------------------------

REAL_DIR.mkdir(parents=True, exist_ok=True)
MANIPULATED_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 70)
print("TRUTHLENS - IMD2020 EXTERNAL DATASET PREPARATION")
print("=" * 70)

print(f"\nSource:")
print(SOURCE)

print(f"\nOutput:")
print(OUTPUT)

# ------------------------------------------------------------
# Find candidate pairs
# ------------------------------------------------------------

pairs = []

for folder in sorted(SOURCE.iterdir()):

    if not folder.is_dir():
        continue

    images = [
        x for x in folder.iterdir()
        if x.is_file()
        and x.suffix.lower() in IMAGE_EXTENSIONS
    ]

    # Find original image
    originals = [
        x for x in images
        if "_orig" in x.stem.lower()
    ]

    if not originals:
        continue

    original = originals[0]

    # Find manipulated images
    manipulated = [
        x for x in images
        if "_orig" not in x.stem.lower()
        and "_mask" not in x.stem.lower()
    ]

    if not manipulated:
        continue

    # Use the first manipulated image from this folder
    manipulated_image = sorted(manipulated)[0]

    pairs.append((folder.name, original, manipulated_image))

# ------------------------------------------------------------
# Show dataset discovery
# ------------------------------------------------------------

print("\nDATASET DISCOVERY")
print("-" * 70)

print(f"Folders containing original + manipulated image: {len(pairs)}")

if len(pairs) < TARGET_PER_CLASS:
    raise ValueError(
        f"Only {len(pairs)} suitable folders found. "
        f"Need at least {TARGET_PER_CLASS}."
    )

# ------------------------------------------------------------
# Select first 100 folders
# ------------------------------------------------------------

selected = pairs[:TARGET_PER_CLASS]

print(f"Folders selected: {len(selected)}")

# ------------------------------------------------------------
# Copy images
# ------------------------------------------------------------

print("\nCOPYING IMAGES")
print("-" * 70)

real_count = 0
manipulated_count = 0

for index, (folder_name, original, manipulated) in enumerate(selected, start=1):

    # Create safe filenames
    real_name = f"imd2020_real_{index:03d}{original.suffix.lower()}"
    manipulated_name = (
        f"imd2020_manipulated_{index:03d}"
        f"{manipulated.suffix.lower()}"
    )

    real_destination = REAL_DIR / real_name
    manipulated_destination = MANIPULATED_DIR / manipulated_name

    shutil.copy2(original, real_destination)
    shutil.copy2(manipulated, manipulated_destination)

    real_count += 1
    manipulated_count += 1

    print(
        f"{index:03d}/100  "
        f"Real: {original.name:<30} "
        f"Manipulated: {manipulated.name}"
    )

# ------------------------------------------------------------
# Final verification
# ------------------------------------------------------------

real_images = [
    x for x in REAL_DIR.iterdir()
    if x.is_file()
    and x.suffix.lower() in IMAGE_EXTENSIONS
]

manipulated_images = [
    x for x in MANIPULATED_DIR.iterdir()
    if x.is_file()
    and x.suffix.lower() in IMAGE_EXTENSIONS
]

print("\n" + "=" * 70)
print("IMD2020 EXTERNAL DATASET READY")
print("=" * 70)

print(f"\nReal images        : {len(real_images)}")
print(f"Manipulated images : {len(manipulated_images)}")
print(f"Total images       : {len(real_images) + len(manipulated_images)}")

print("\nLocation:")
print(OUTPUT)

print("\nExpected structure:")
print(OUTPUT / "Real")
print(OUTPUT / "Manipulated")

print("\nNext step:")
print("Run the IMD2020 external validation script.")

print("=" * 70)