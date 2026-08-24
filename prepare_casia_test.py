from pathlib import Path
import shutil


# ============================================================
# TRUTHLENS - CASIA 2.0 TEST SET PREPARATION
# ============================================================

PROJECT_DIR = Path(r"D:\TruthLens")

CASIA_DIR = Path(r"D:\CASIA2")

SOURCE_DIR = CASIA_DIR / "Tp"

OUTPUT_DIR = (
    PROJECT_DIR /
    "RealWorld_CASIA_Test" /
    "Manipulated"
)

NUMBER_OF_IMAGES = 100



# ============================================================
# CHECK SOURCE
# ============================================================

print("=" * 70)
print("TRUTHLENS - CASIA 2.0 TEST SET PREPARATION")
print("=" * 70)

print("\nCASIA source:")
print(SOURCE_DIR)

print("\nOutput:")
print(OUTPUT_DIR)


if not SOURCE_DIR.exists():

    raise FileNotFoundError(
        "\nCASIA tampered-image folder was not found:\n"
        f"{SOURCE_DIR}\n\n"
        "Check the CASIA2 folder location."
    )


# ============================================================
# CREATE OUTPUT
# ============================================================

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# FIND IMAGES
# ============================================================

extensions = [
    "*.jpg",
    "*.jpeg",
    "*.png",
    "*.bmp",
    "*.tif",
    "*.tiff"
]

images = []

for extension in extensions:

    images.extend(
        SOURCE_DIR.glob(extension)
    )


images = sorted(images)


print(
    f"\nTampered images found: "
    f"{len(images)}"
)


if len(images) < NUMBER_OF_IMAGES:

    raise RuntimeError(
        "Not enough tampered images found."
    )


# ============================================================
# COPY FIRST 100
# ============================================================

print(
    f"\nSelecting {NUMBER_OF_IMAGES} "
    "tampered images..."
)


selected = images[:NUMBER_OF_IMAGES]


for index, image_path in enumerate(
    selected,
    start=1
):

    output_path = (
        OUTPUT_DIR /
        f"casia_manipulated_{index:03d}.jpg"
    )

    try:

        shutil.copy2(
            image_path,
            output_path
        )

        print(
            f"{index:03d}/"
            f"{NUMBER_OF_IMAGES}  "
            f"{image_path.name}"
        )

    except Exception as error:

        print(
            f"ERROR: {image_path.name}"
        )

        print(error)


# ============================================================
# FINAL
# ============================================================

final_count = len([
    file
    for file in OUTPUT_DIR.iterdir()
    if file.is_file()
])


print("\n" + "=" * 70)
print("CASIA TEST SET READY")
print("=" * 70)

print(
    f"\nImages copied: {final_count}"
)

print(
    "\nLocation:"
)

print(OUTPUT_DIR)

print("\nNext step:")

print(
    "Run the CASIA validation script."
)