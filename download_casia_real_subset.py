import shutil
from pathlib import Path

# ============================================================
# TRUTHLENS - CASIA 2.0 AUTHENTIC IMAGE SUBSET
# ============================================================

CASIA_AU_DIR = Path(r"D:\CASIA2\Au")

OUTPUT_DIR = Path(
    r"D:\TruthLens\RealWorld_CASIA_Test\Real"
)

NUM_IMAGES = 100

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 70)
print("TRUTHLENS - CASIA 2.0 AUTHENTIC SUBSET")
print("=" * 70)

print("\nSource:")
print(CASIA_AU_DIR)

print("\nDestination:")
print(OUTPUT_DIR)

# ------------------------------------------------------------
# Find images
# ------------------------------------------------------------

extensions = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff"
}

images = []

for file in CASIA_AU_DIR.rglob("*"):

    if file.is_file() and file.suffix.lower() in extensions:
        images.append(file)

print(f"\nAuthentic images found: {len(images)}")

if len(images) < NUM_IMAGES:
    raise ValueError(
        f"Only {len(images)} images found. "
        f"Need at least {NUM_IMAGES}."
    )

# ------------------------------------------------------------
# Copy first 100 images
# ------------------------------------------------------------

images = images[:NUM_IMAGES]

print("\nCopying images...\n")

for index, image_path in enumerate(images, start=1):

    output_name = (
        f"casia_real_{index:03d}"
        + image_path.suffix.lower()
    )

    destination = OUTPUT_DIR / output_name

    shutil.copy2(
        image_path,
        destination
    )

    print(
        f"{index:03d}/{NUM_IMAGES}  "
        f"{image_path.name} -> {output_name}"
    )

print("\n" + "=" * 70)
print("CASIA AUTHENTIC TEST SET READY")
print("=" * 70)

print(f"\nImages copied: {len(images)}")

print("\nLocation:")
print(OUTPUT_DIR)

print("\nNext step:")
print(
    "Run the CASIA authentic validation script."
)