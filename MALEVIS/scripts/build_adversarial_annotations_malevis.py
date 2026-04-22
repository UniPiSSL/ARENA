from pathlib import Path
import pandas as pd

ROOT = Path.home() / "malevis_adversarial_repro"
CLEAN_CSV = ROOT / "data" / "annotations" / "new_annotations.csv"
ADV_ROOT = ROOT / "artifacts" / "adv_outputs"
OUT_DIR = ROOT / "data" / "annotations" / "adv_annotations"
OUT_DIR.mkdir(parents=True, exist_ok=True)

ATTACK_FOLDERS = [
    "ANN_fgsm",
    "ANN_pgd",
    "CNN_fgsm",
    "CNN_pgd",
    "RNN_fgsm",
    "RNN_pgd",
]

def main():
    clean_df = pd.read_csv(CLEAN_CSV)

    required_cols = ["image_id", "label", "isNight", "split", "class_name"]
    missing = [c for c in required_cols if c not in clean_df.columns]
    if missing:
        raise ValueError(f"Missing columns in clean CSV: {missing}")

    summary_rows = []

    for folder in ATTACK_FOLDERS:
        adv_df = clean_df.copy()

        def to_adv_path(orig_path, split, class_name):
            fname = Path(orig_path).name
            return str(ADV_ROOT / folder / split / class_name / fname)

        adv_df["image_id"] = adv_df.apply(
            lambda r: to_adv_path(r["image_id"], r["split"], r["class_name"]),
            axis=1
        )

        # sanity: keep only rows whose generated file actually exists
        adv_df["exists"] = adv_df["image_id"].apply(lambda p: Path(p).exists())
        missing_count = (~adv_df["exists"]).sum()
        if missing_count > 0:
            print(f"[WARN] {folder}: dropping {missing_count} missing files")
        adv_df = adv_df[adv_df["exists"]].drop(columns=["exists"]).reset_index(drop=True)

        out_csv = OUT_DIR / f"Annotations_{folder}.csv"
        adv_df.to_csv(out_csv, index=False)

        summary_rows.append({
            "attack_folder": folder,
            "rows": len(adv_df),
            "csv_path": str(out_csv),
            "train_rows": int((adv_df["split"] == "train").sum()),
            "val_rows": int((adv_df["split"] == "val").sum()),
            "test_rows": int((adv_df["split"] == "test").sum()),
        })

        print(f"Saved: {out_csv} ({len(adv_df)} rows)")

    summary_df = pd.DataFrame(summary_rows)
    summary_path = OUT_DIR / "adv_annotation_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"\nSaved summary: {summary_path}")
    print(summary_df)

if __name__ == "__main__":
    main()
