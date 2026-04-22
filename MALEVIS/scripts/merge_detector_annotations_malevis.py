from pathlib import Path
import pandas as pd

ROOT = Path.home() / "malevis_adversarial_repro"
CLEAN_CSV = ROOT / "data" / "annotations" / "new_annotations.csv"
ADV_DIR = ROOT / "data" / "annotations" / "adv_annotations"
OUT_DIR = ROOT / "data" / "annotations" / "detector_annotations"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Known / unknown setup
KNOWN_ATTACK_CSV = ADV_DIR / "Annotations_CNN_fgsm.csv"
UNKNOWN_FGSM_CSV = ADV_DIR / "Annotations_RNN_fgsm.csv"
UNKNOWN_PGD_CSV = ADV_DIR / "Annotations_CNN_pgd.csv"

def make_detector_df(clean_df, adv_df, split_clean, split_adv):
    clean_part = clean_df[clean_df["split"] == split_clean].copy()
    adv_part = adv_df[adv_df["split"] == split_adv].copy()

    clean_part["detector_label"] = 0
    adv_part["detector_label"] = 1

    clean_part["source"] = "clean"
    adv_part["source"] = "adversarial"

    merged = pd.concat([clean_part, adv_part], ignore_index=True)
    return merged

def save_csv(df, path):
    df.to_csv(path, index=False)
    print(f"Saved: {path} ({len(df)} rows)")

def main():
    clean_df = pd.read_csv(CLEAN_CSV)
    known_df = pd.read_csv(KNOWN_ATTACK_CSV)
    unknown_fgsm_df = pd.read_csv(UNKNOWN_FGSM_CSV)
    unknown_pgd_df = pd.read_csv(UNKNOWN_PGD_CSV)

    # 1) Train detector annotations (known attack)
    train_detector_df = make_detector_df(clean_df, known_df, split_clean="train", split_adv="train")
    save_csv(train_detector_df, OUT_DIR / "train_detector_annotations.csv")

    # 2) Eval detector annotations (known attack on test)
    eval_detector_df = make_detector_df(clean_df, known_df, split_clean="test", split_adv="test")
    save_csv(eval_detector_df, OUT_DIR / "eval_detector_annotations.csv")

    # 3) Unknown unseen FGSM (RNN source)
    unknown_fgsm_eval_df = make_detector_df(clean_df, unknown_fgsm_df, split_clean="test", split_adv="test")
    save_csv(unknown_fgsm_eval_df, OUT_DIR / "unknown_adversarial_fgsm_RNN_detector_annotations.csv")

    # 4) Unknown unseen PGD (CNN source)
    unknown_pgd_eval_df = make_detector_df(clean_df, unknown_pgd_df, split_clean="test", split_adv="test")
    save_csv(unknown_pgd_eval_df, OUT_DIR / "unknown_adversarial_PGD_detector_annotations.csv")

    # 5) Semi-supervised detector annotations (same training pool for now)
    semi_df = make_detector_df(clean_df, known_df, split_clean="train", split_adv="train")
    save_csv(semi_df, OUT_DIR / "train_semisupervised_detector_annotations.csv")

    # summary
    summary_rows = []
    for name, df in [
        ("train_detector_annotations", train_detector_df),
        ("eval_detector_annotations", eval_detector_df),
        ("unknown_adversarial_fgsm_RNN_detector_annotations", unknown_fgsm_eval_df),
        ("unknown_adversarial_PGD_detector_annotations", unknown_pgd_eval_df),
        ("train_semisupervised_detector_annotations", semi_df),
    ]:
        summary_rows.append({
            "name": name,
            "rows": len(df),
            "clean_rows": int((df["detector_label"] == 0).sum()),
            "adv_rows": int((df["detector_label"] == 1).sum()),
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_path = OUT_DIR / "detector_annotation_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"\nSaved summary: {summary_path}")
    print(summary_df)

if __name__ == "__main__":
    main()
