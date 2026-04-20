import os
import shutil
import cv2
import numpy as np

# ----------------------------
# Config
# ----------------------------
images_folder = "CrackForest/Images"
masks_folder  = "CrackForest/Masks"

output_crack    = "dataset/crack"
output_no_crack = "dataset/no_crack"

# Crack severity thresholds (white pixel count in mask)
THRESHOLD_NO_CRACK  = 100    # below this → no crack
THRESHOLD_CRACK     = 100    # above this → crack

os.makedirs(output_crack, exist_ok=True)
os.makedirs(output_no_crack, exist_ok=True)

# ----------------------------
# Stats tracking
# ----------------------------
total       = 0
crack_count = 0
no_crack_count = 0
skipped     = 0

severity_buckets = {"low": 0, "medium": 0, "high": 0}  # crack only

# ----------------------------
# Process each image
# ----------------------------
for filename in sorted(os.listdir(images_folder)):
    if not filename.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
        continue

    image_path = os.path.join(images_folder, filename)
    name = os.path.splitext(filename)[0]

    # CrackForest mask format: 110_label.PNG
    mask_name = name + "_label.PNG"
    mask_path = os.path.join(masks_folder, mask_name)

    if not os.path.exists(mask_path):
        print(f"[SKIP] Mask not found: {filename}")
        skipped += 1
        continue

    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        print(f"[SKIP] Could not read mask: {filename}")
        skipped += 1
        continue

    # Compute crack pixel ratio
    total_pixels  = mask.shape[0] * mask.shape[1]
    white_pixels  = int(np.sum(mask > 10))
    crack_ratio   = white_pixels / total_pixels

    total += 1

    if white_pixels > THRESHOLD_CRACK:
        shutil.copy(image_path, os.path.join(output_crack, filename))
        crack_count += 1

        # Severity grading
        if crack_ratio < 0.05:
            severity_buckets["low"] += 1
        elif crack_ratio < 0.15:
            severity_buckets["medium"] += 1
        else:
            severity_buckets["high"] += 1

    else:
        shutil.copy(image_path, os.path.join(output_no_crack, filename))
        no_crack_count += 1

# ----------------------------
# Summary
# ----------------------------
print("\n========== Dataset Preparation Complete ==========")
print(f"Total images processed : {total}")
print(f"  Crack images         : {crack_count}")
print(f"    └─ Low severity    : {severity_buckets['low']}")
print(f"    └─ Medium severity : {severity_buckets['medium']}")
print(f"    └─ High severity   : {severity_buckets['high']}")
print(f"  No-crack images      : {no_crack_count}")
print(f"  Skipped (no mask)    : {skipped}")
print(f"\nClass balance ratio    : {crack_count}:{no_crack_count} (crack:no_crack)")

if crack_count > 0 and no_crack_count > 0:
    ratio = max(crack_count, no_crack_count) / min(crack_count, no_crack_count)
    if ratio > 2.0:
        print(f"\n[WARNING] Imbalanced dataset detected (ratio {ratio:.1f}x).")
        print("  Consider using class_weight in model.fit() or oversampling the minority class.")
    else:
        print(f"\n[OK] Dataset is reasonably balanced (ratio {ratio:.1f}x).")

print(f"\nOutput folders:")
print(f"  Crack    → {output_crack}/")
print(f"  No crack → {output_no_crack}/")
print("==================================================\n")