import os
from PIL import Image

# ==============================
# TruthLens Image Verification
# ==============================

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_PATH = os.path.join(PROJECT_ROOT, "Dataset")

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")


datasets = {
    "COCO (Real)": os.path.join(DATASET_PATH, "Real", "COCO", "train2017"),

    "TinyGenImage (AI)": os.path.join(DATASET_PATH, "AI_Generated", "TinyGenImage"),

    "CASIA Original (Au)": os.path.join(DATASET_PATH, "Manipulated", "CASIA", "CASIA2", "Au"),

    "CASIA Tampered (Tp)": os.path.join(DATASET_PATH, "Manipulated", "CASIA", "CASIA2", "Tp"),

    "FaceForensics": os.path.join(DATASET_PATH, "Deepfake", "FaceForensics", "cropped_images"),
}


def get_total_images(folder):

    total = 0

    for root, dirs, files in os.walk(folder):
        total += sum(
            1 for f in files
            if f.lower().endswith(IMAGE_EXTENSIONS)
        )

    return total


def progress_bar(progress, total, length=30):

    percent = progress / total

    filled = int(length * percent)

    bar = "█" * filled + "-" * (length - filled)

    return f"[{bar}] {percent*100:6.2f}%"


print("=" * 75)
print("TruthLens Image Verification")
print("=" * 75)

overall_good = 0
overall_bad = 0

for dataset_name, folder in datasets.items():

    if not os.path.exists(folder):
        print(f"\n{dataset_name} : Folder not found.")
        continue

    total_images = get_total_images(folder)

    print(f"\nChecking {dataset_name}")
    print(f"Total Images : {total_images}\n")

    checked = 0
    good = 0
    bad = 0

    for root, dirs, files in os.walk(folder):

        for file in files:

            if not file.lower().endswith(IMAGE_EXTENSIONS):
                continue

            path = os.path.join(root, file)

            try:
                with Image.open(path) as img:
                    img.verify()

                good += 1

            except Exception:
                bad += 1

            checked += 1

            # Update every 500 images
            if checked % 500 == 0 or checked == total_images:

                print(
                    "\r"
                    + progress_bar(checked, total_images)
                    + f"   Checked:{checked:,}"
                    + f"   Good:{good:,}"
                    + f"   Bad:{bad:,}",
                    end=""
                )

    overall_good += good
    overall_bad += bad

    print("\n")
    print("-" * 75)
    print(f"{dataset_name} Finished")
    print(f"Good Images      : {good:,}")
    print(f"Corrupted Images : {bad:,}")
    print("-" * 75)

print("\n")
print("=" * 75)
print("Verification Completed")
print("=" * 75)

print(f"Total Good Images      : {overall_good:,}")
print(f"Total Corrupted Images : {overall_bad:,}")