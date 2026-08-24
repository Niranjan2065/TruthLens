from pathlib import Path
from PIL import Image, ImageEnhance, ImageFilter
import random
import shutil

# ============================================================
# TRUTHLENS - REAL-WORLD TRAINING DATA PREPARATION
# ============================================================

SOURCE_DIR = Path(r"D:\TruthLens\Final_Dataset\Train")
OUTPUT_DIR = Path(r"D:\TruthLens\RealWorld_Training")

CLASSES = [
    "AI_Generated",
    "Deepfake",
    "Manipulated",
    "Real"
]

TARGET_PER_CLASS = 500
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

RANDOM_SEED = 42
random.seed(RANDOM_SEED)


# ============================================================
# FUNCTIONS
# ============================================================

def get_images(folder):
    """Return valid image files from a folder."""
    return [
        p for p in folder.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    ]


def screen_capture_transform(img, index):
    """
    Simulate characteristics commonly introduced when
    digital media is viewed and captured from a phone screen.

    The goal is NOT to create fake content.
    The goal is to make the classifier more robust to
    real-world screen-capture conditions.
    """

    img = img.convert("RGB")

    # --------------------------------------------------------
    # 1. Resize while preserving aspect ratio
    # --------------------------------------------------------
    max_side = 900

    width, height = img.size

    scale = min(max_side / width, max_side / height, 1.0)

    new_width = max(1, int(width * scale))
    new_height = max(1, int(height * scale))

    img = img.resize(
        (new_width, new_height),
        Image.Resampling.LANCZOS
    )

    # --------------------------------------------------------
    # 2. Mild brightness variation
    # --------------------------------------------------------
    brightness = random.uniform(0.92, 1.08)
    img = ImageEnhance.Brightness(img).enhance(brightness)

    # --------------------------------------------------------
    # 3. Mild contrast variation
    # --------------------------------------------------------
    contrast = random.uniform(0.92, 1.08)
    img = ImageEnhance.Contrast(img).enhance(contrast)

    # --------------------------------------------------------
    # 4. Very small blur to simulate capture/compression
    # --------------------------------------------------------
    if random.random() < 0.35:
        img = img.filter(
            ImageFilter.GaussianBlur(
                radius=random.uniform(0.1, 0.4)
            )
        )

    # --------------------------------------------------------
    # 5. JPEG compression
    # --------------------------------------------------------
    # Save/reload through JPEG later to simulate
    # social-media/screenshot compression.
    return img


def save_image(img, output_path, quality):
    """Save transformed image as JPEG."""
    img.save(
        output_path,
        format="JPEG",
        quality=quality,
        optimize=True
    )


# ============================================================
# START
# ============================================================

print("=" * 70)
print("TRUTHLENS - REAL-WORLD TRAINING DATA PREPARATION")
print("=" * 70)

print()
print("Source:")
print(SOURCE_DIR)

print()
print("Output:")
print(OUTPUT_DIR)

print()
print("Target per class:", TARGET_PER_CLASS)

# ------------------------------------------------------------
# Check source
# ------------------------------------------------------------

if not SOURCE_DIR.exists():
    raise FileNotFoundError(
        f"Source dataset not found:\n{SOURCE_DIR}"
    )

# ------------------------------------------------------------
# Create output folders
# ------------------------------------------------------------

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

for class_name in CLASSES:
    (OUTPUT_DIR / class_name).mkdir(
        parents=True,
        exist_ok=True
    )

# ============================================================
# PROCESS EACH CLASS
# ============================================================

total_created = 0

print()
print("=" * 70)
print("DATASET PREPARATION")
print("=" * 70)

for class_name in CLASSES:

    source_class = SOURCE_DIR / class_name
    output_class = OUTPUT_DIR / class_name

    print()
    print("-" * 70)
    print("CLASS:", class_name)
    print("-" * 70)

    if not source_class.exists():
        print("WARNING: Source folder not found:")
        print(source_class)
        continue

    images = get_images(source_class)

    print("Available source images:", len(images))

    if len(images) == 0:
        print("WARNING: No images found.")
        continue

    # --------------------------------------------------------
    # Randomly select source images
    # --------------------------------------------------------

    if len(images) >= TARGET_PER_CLASS:
        selected = random.sample(
            images,
            TARGET_PER_CLASS
        )
    else:
        print(
            f"Only {len(images)} images available. "
            f"Some images will be reused with different transformations."
        )

        selected = [
            random.choice(images)
            for _ in range(TARGET_PER_CLASS)
        ]

    created = 0

    # --------------------------------------------------------
    # Process selected images
    # --------------------------------------------------------

    for i, image_path in enumerate(selected, start=1):

        try:

            img = Image.open(image_path)

            # Convert to RGB
            img = img.convert("RGB")

            # Apply real-world/screen-capture style transformation
            transformed = screen_capture_transform(
                img,
                i
            )

            # Random JPEG quality
            jpeg_quality = random.randint(70, 95)

            output_name = (
                f"{class_name.lower()}_realworld_"
                f"{i:04d}.jpg"
            )

            output_path = output_class / output_name

            save_image(
                transformed,
                output_path,
                jpeg_quality
            )

            created += 1
            total_created += 1

            if created % 50 == 0:
                print(
                    f"Created: {created}/{TARGET_PER_CLASS}"
                )

        except Exception as e:

            print(
                f"WARNING: Could not process "
                f"{image_path.name}"
            )

            print("Reason:", e)

    print()
    print(
        f"{class_name}: {created} images created"
    )


# ============================================================
# FINAL REPORT
# ============================================================

print()
print("=" * 70)
print("REAL-WORLD DATASET PREPARATION COMPLETE")
print("=" * 70)

print()

grand_total = 0

for class_name in CLASSES:

    folder = OUTPUT_DIR / class_name

    count = len([
        p for p in folder.iterdir()
        if p.is_file()
        and p.suffix.lower() in IMAGE_EXTENSIONS
    ])

    grand_total += count

    print(
        f"{class_name:<15}: {count:>5} images"
    )

print("-" * 70)
print(
    f"{'TOTAL':<15}: {grand_total:>5} images"
)

print()
print("Output location:")
print(OUTPUT_DIR)

print()
print("Expected structure:")
print(
    r"D:\TruthLens\RealWorld_Training\AI_Generated"
)
print(
    r"D:\TruthLens\RealWorld_Training\Deepfake"
)
print(
    r"D:\TruthLens\RealWorld_Training\Manipulated"
)
print(
    r"D:\TruthLens\RealWorld_Training\Real"
)

print()
print("=" * 70)
print("NEXT STEP: INSPECT DATASET COUNTS BEFORE TRAINING")
print("=" * 70)