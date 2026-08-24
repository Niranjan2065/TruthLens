from pathlib import Path
from collections import Counter

ROOT = Path(r"D:\IMD2020")

IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"
}

print("=" * 70)
print("TRUTHLENS - IMD2020 DATASET INSPECTION")
print("=" * 70)

print(f"\nDataset location:")
print(ROOT)

# --------------------------------------------------
# 1. Count folders
# --------------------------------------------------

folders = [x for x in ROOT.iterdir() if x.is_dir()]

print(f"\nTop-level folders : {len(folders)}")

# --------------------------------------------------
# 2. Count images
# --------------------------------------------------

images = [
    x for x in ROOT.rglob("*")
    if x.is_file() and x.suffix.lower() in IMAGE_EXTENSIONS
]

print(f"Total images      : {len(images)}")

# --------------------------------------------------
# 3. Image extensions
# --------------------------------------------------

extensions = Counter(x.suffix.lower() for x in images)

print("\nIMAGE EXTENSIONS")
print("-" * 40)

for ext, count in sorted(extensions.items()):
    print(f"{ext:10} : {count}")

# --------------------------------------------------
# 4. Inspect first 10 folders
# --------------------------------------------------

print("\nFIRST 10 FOLDERS")
print("-" * 40)

for folder in folders[:10]:

    folder_images = [
        x for x in folder.rglob("*")
        if x.is_file() and x.suffix.lower() in IMAGE_EXTENSIONS
    ]

    print(f"{folder.name:15} : {len(folder_images)} images")

    for img in folder_images[:3]:
        print(f"    {img.name}")

# --------------------------------------------------
# 5. Search for metadata / label files
# --------------------------------------------------

print("\nPOSSIBLE METADATA FILES")
print("-" * 40)

metadata_extensions = {
    ".txt", ".csv", ".json", ".xml", ".mat",
    ".tsv", ".xls", ".xlsx"
}

metadata_files = [
    x for x in ROOT.rglob("*")
    if x.is_file() and x.suffix.lower() in metadata_extensions
]

print(f"Metadata files found: {len(metadata_files)}")

for file in metadata_files[:30]:
    print(file)

# --------------------------------------------------
# 6. Search filenames for useful labels
# --------------------------------------------------

print("\nLABEL KEYWORD CHECK")
print("-" * 40)

keywords = [
    "real",
    "authentic",
    "original",
    "fake",
    "forged",
    "manipulated",
    "tampered",
    "spliced"
]

keyword_counts = Counter()

for img in images:

    name = img.name.lower()

    for keyword in keywords:
        if keyword in name:
            keyword_counts[keyword] += 1

for keyword, count in keyword_counts.items():
    print(f"{keyword:15} : {count}")

print("\n" + "=" * 70)
print("IMD2020 INSPECTION COMPLETE")
print("=" * 70)