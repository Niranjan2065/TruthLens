import os

# Dataset paths
DATASETS = {
    "COCO (Real)": r"D:\TruthLens\Dataset\Real\COCO\train2017",
    "TinyGenImage": r"D:\TruthLens\Dataset\AI_Generated\TinyGenImage",
    "CASIA Original (Au)": r"D:\TruthLens\Dataset\Manipulated\CASIA\CASIA2\Au",
    "CASIA Tampered (Tp)": r"D:\TruthLens\Dataset\Manipulated\CASIA\CASIA2\Tp",
    "FaceForensics": r"D:\TruthLens\Dataset\Deepfake\FaceForensics\cropped_images"
}

IMAGE_EXTENSIONS = (
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff"
)

print("=" * 70)
print("TruthLens Dataset Verification")
print("=" * 70)

grand_total = 0

for dataset_name, dataset_path in DATASETS.items():

    total = 0

    for root, dirs, files in os.walk(dataset_path):
        total += sum(
            1 for file in files
            if file.lower().endswith(IMAGE_EXTENSIONS)
        )

    grand_total += total

    print(f"{dataset_name:<30} : {total:,} images")

print("-" * 70)
print(f"{'TOTAL':<30} : {grand_total:,} images")
print("=" * 70)