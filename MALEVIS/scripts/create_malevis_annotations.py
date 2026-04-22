import csv
import random
from pathlib import Path

SEED = 42
VAL_TEST_RATIO = 0.5  # split original val into new val/test
ROOT = Path.home() / "malevis_adversarial_repro"
DATASET_ROOT = ROOT / "data" / "raw" / "malevis_train_val_300x300"
OUT_DIR = ROOT / "data" / "annotations"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_DIR = DATASET_ROOT / "train"
VAL_DIR = DATASET_ROOT / "val"

random.seed(SEED)

# 1) Discover classes from train folder in alphabetical order
classes = sorted([p.name for p in TRAIN_DIR.iterdir() if p.is_dir()])

# Safety checks
val_classes = sorted([p.name for p in VAL_DIR.iterdir() if p.is_dir()])
if classes != val_classes:
    raise RuntimeError(
        f"Train/val class mismatch.\nTrain: {classes}\nVal:   {val_classes}"
    )

# 2) Build mappings
# Keep CSV labels as 1..N to stay close to original pipeline expectations
class_to_csv_label = {cls_name: i + 1 for i, cls_name in enumerate(classes)}

# Also save zero-based for reference/debug
class_to_zero_label = {cls_name: i for i, cls_name in enumerate(classes)}

# 3) Save label map
label_map_path = OUT_DIR / "label_map.csv"
with open(label_map_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["class_name", "csv_label_1based", "zero_based_label"])
    for cls_name in classes:
        writer.writerow([cls_name, class_to_csv_label[cls_name], class_to_zero_label[cls_name]])

# 4) Build annotation rows
rows = []

# Train stays train
for cls_name in classes:
    files = sorted((TRAIN_DIR / cls_name).glob("*.png"))
    for fp in files:
        rows.append([
            str(fp),
            class_to_csv_label[cls_name],  # 1..26
            0,                             # isNight
            "train",
            cls_name,
        ])

# Original val gets split into new val/test stratified per class
for cls_name in classes:
    files = sorted((VAL_DIR / cls_name).glob("*.png"))
    files = list(files)
    rng = random.Random(SEED + class_to_zero_label[cls_name])
    rng.shuffle(files)

    n_total = len(files)
    n_val = int(round(n_total * VAL_TEST_RATIO))
    n_val = max(1, min(n_total - 1, n_val))  # ensure both val/test non-empty

    new_val_files = files[:n_val]
    test_files = files[n_val:]

    for fp in new_val_files:
        rows.append([
            str(fp),
            class_to_csv_label[cls_name],
            0,
            "val",
            cls_name,
        ])

    for fp in test_files:
        rows.append([
            str(fp),
            class_to_csv_label[cls_name],
            0,
            "test",
            cls_name,
        ])

# 5) Save annotations
ann_path = OUT_DIR / "new_annotations.csv"
with open(ann_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["image_id", "label", "isNight", "split", "class_name"])
    writer.writerows(rows)

# 6) Print summary
summary = {}
for _, label, _, split, cls_name in rows:
    summary.setdefault(split, {})
    summary[split].setdefault(cls_name, 0)
    summary[split][cls_name] += 1

print(f"Saved: {ann_path}")
print(f"Saved: {label_map_path}")
print(f"Classes ({len(classes)}): {classes}")

for split in ["train", "val", "test"]:
    total = sum(summary.get(split, {}).values())
    print(f"\n[{split}] total = {total}")
    for cls_name in classes:
        print(f"  {cls_name}: {summary.get(split, {}).get(cls_name, 0)}")
