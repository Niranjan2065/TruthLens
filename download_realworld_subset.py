from pathlib import Path
from datasets import load_dataset
from PIL import Image
import io


# ============================================================
# TRUTHLENS - AUTOMATIC REAL-WORLD VALIDATION DATASET
# ============================================================

PROJECT_DIR = Path(r"D:\TruthLens")
OUTPUT_DIR = PROJECT_DIR / "RealWorld_Test"

LIMIT = 100


# ============================================================
# FOLDERS
# ============================================================

FOLDERS = {
    "AI_Generated": OUTPUT_DIR / "AI_Generated",
    "Deepfake": OUTPUT_DIR / "Deepfake",
    "Manipulated": OUTPUT_DIR / "Manipulated",
    "Real": OUTPUT_DIR / "Real",
}


for folder in FOLDERS.values():
    folder.mkdir(
        parents=True,
        exist_ok=True
    )


# ============================================================
# HELPER
# ============================================================

def save_image(image, output_path):

    if not isinstance(image, Image.Image):
        image = Image.open(
            io.BytesIO(image)
        )

    image = image.convert("RGB")

    image.save(
        output_path,
        format="JPEG",
        quality=95
    )


def existing_count(folder):

    extensions = (
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp",
        ".webp"
    )

    return len([
        p for p in folder.iterdir()
        if p.is_file()
        and p.suffix.lower() in extensions
    ])


# ============================================================
# 1. AI-GENERATED
# ============================================================

def download_ai_generated():

    folder = FOLDERS["AI_Generated"]

    current = existing_count(folder)

    if current >= LIMIT:
        print(
            f"AI_Generated already has "
            f"{current} images."
        )
        return

    print("\n" + "=" * 70)
    print("DOWNLOADING AI-GENERATED IMAGES")
    print("=" * 70)

    dataset = load_dataset(
        "TheKernel01/Tiny-GenImage",
        split="train",
        streaming=True
    )

    count = current

    for item in dataset:

        # label 1 = AI generated
        if item["label"] != 1:
            continue

        image = item["image"]

        output_path = (
            folder /
            f"ai_generated_{count + 1:03d}.jpg"
        )

        save_image(
            image,
            output_path
        )

        count += 1

        print(
            f"AI_Generated: "
            f"{count}/{LIMIT}"
        )

        if count >= LIMIT:
            break

    print(
        f"AI_Generated complete: "
        f"{count} images"
    )


# ============================================================
# 2. DEEPFAKE
# ============================================================

def download_deepfake():

    folder = FOLDERS["Deepfake"]

    current = existing_count(folder)

    if current >= LIMIT:
        print(
            f"Deepfake already has "
            f"{current} images."
        )
        return

    print("\n" + "=" * 70)
    print("DOWNLOADING DEEPFAKE IMAGES")
    print("=" * 70)

    dataset = load_dataset(
        "ulasbngl/WildDeepfake_subset_large",
        split="train",
        streaming=True
    )

    count = current

    for item in dataset:

        # label 1 = fake/deepfake
        if int(item["label"]) != 1:
            continue

        image = item["png"]

        output_path = (
            folder /
            f"deepfake_{count + 1:03d}.jpg"
        )

        save_image(
            image,
            output_path
        )

        count += 1

        print(
            f"Deepfake: "
            f"{count}/{LIMIT}"
        )

        if count >= LIMIT:
            break

    print(
        f"Deepfake complete: "
        f"{count} images"
    )


# ============================================================
# 3. MANIPULATED
# ============================================================

def download_manipulated():

    folder = FOLDERS["Manipulated"]

    current = existing_count(folder)

    if current >= LIMIT:
        print(
            f"Manipulated already has "
            f"{current} images."
        )
        return

    print("\n" + "=" * 70)
    print("DOWNLOADING MANIPULATED IMAGES")
    print("=" * 70)

    dataset = load_dataset(
        "FatimahEmadEldin/genai-manipulation-detection-interior",
        split="train",
        streaming=True
    )

    count = current

    for item in dataset:

        # Dataset label:
        # 0 = fake/manipulated
        if item["label"] != 0:
            continue

        image = item["image"]

        output_path = (
            folder /
            f"manipulated_{count + 1:03d}.jpg"
        )

        save_image(
            image,
            output_path
        )

        count += 1

        print(
            f"Manipulated: "
            f"{count}/{LIMIT}"
        )

        if count >= LIMIT:
            break

    print(
        f"Manipulated complete: "
        f"{count} images"
    )


# ============================================================
# 4. REAL
# ============================================================

def download_real():

    folder = FOLDERS["Real"]

    current = existing_count(folder)

    if current >= LIMIT:
        print(
            f"Real already has "
            f"{current} images."
        )
        return

    print("\n" + "=" * 70)
    print("DOWNLOADING REAL IMAGES")
    print("=" * 70)

    dataset = load_dataset(
        "regisss/coco_2017",
        split="validation",
        streaming=True
    )

    count = current

    for item in dataset:

        image = item["image"]

        if image is None:
            continue

        output_path = (
            folder /
            f"real_{count + 1:03d}.jpg"
        )

        save_image(
            image,
            output_path
        )

        count += 1

        print(
            f"Real: "
            f"{count}/{LIMIT}"
        )

        if count >= LIMIT:
            break

    print(
        f"Real complete: "
        f"{count} images"
    )


# ============================================================
# MAIN
# ============================================================

print("=" * 70)
print("TRUTHLENS REAL-WORLD DATASET BUILDER")
print("=" * 70)

print(
    f"\nOutput directory:\n{OUTPUT_DIR}"
)

print(
    f"\nTarget: {LIMIT} images per class"
)


download_ai_generated()

download_deepfake()

download_manipulated()

download_real()


# ============================================================
# FINAL CHECK
# ============================================================

print("\n" + "=" * 70)
print("DATASET DOWNLOAD COMPLETE")
print("=" * 70)

for name, folder in FOLDERS.items():

    count = existing_count(folder)

    print(
        f"{name:15} : {count} images"
    )

print("\nLocation:")

print(OUTPUT_DIR)

print("\nNext step:")

print(
    "python D:\\TruthLens\\realworld_validation.py"
)