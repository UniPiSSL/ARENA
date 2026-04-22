import os
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
import tensorflow as tf

from tensorflow.keras.models import load_model

from art.estimators.classification import TensorFlowV2Classifier
from art.attacks.evasion import FastGradientMethod, ProjectedGradientDescent

# =========================
# Config
# =========================
IMG_SIZE = 50
NUM_CLASSES = 26

ROOT = Path.home() / "malevis_adversarial_repro"
CSV_PATH = ROOT / "data" / "annotations" / "new_annotations.csv"
MODELS_DIR = ROOT / "artifacts" / "models"
ADV_ROOT = ROOT / "artifacts" / "adv_outputs"
LOGS_DIR = ROOT / "artifacts" / "logs"

ADV_ROOT.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)

# Attack settings
FGSM_EPS = 8 / 255.0
PGD_EPS = 8 / 255.0
PGD_EPS_STEP = 2 / 255.0
PGD_MAX_ITER = 10

# Generate all source-model / attack combinations
ATTACK_PLAN = {
    "ANN": ["fgsm", "pgd"],
    "CNN": ["fgsm", "pgd"],
    "RNN": ["fgsm", "pgd"],
}

TARGET_SPLITS = ["train", "val", "test"]


# =========================
# Helpers
# =========================
def load_annotations():
    df = pd.read_csv(CSV_PATH)
    df["label"] = df["label"].astype(int) - 1
    return df


def one_hot(labels, num_classes):
    return np.eye(num_classes, dtype=np.float32)[labels]


def load_image(img_path):
    img = cv2.imread(str(img_path))
    if img is None:
        raise ValueError(f"Could not read image: {img_path}")
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE)).astype("float32") / 255.0
    return img


def save_adv_png(x_adv, out_path):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img = np.clip(x_adv * 255.0, 0, 255).astype(np.uint8)
    cv2.imwrite(str(out_path), img)


def ann_preprocess(img, is_night):
    x_img = img.reshape(1, -1).astype("float32")
    x_isn = np.array([[is_night]], dtype="float32")
    x = np.hstack([x_img, x_isn]).astype("float32")
    return x


def ann_postprocess(x_adv):
    return x_adv[0, :-1].reshape(IMG_SIZE, IMG_SIZE, 3)


def cnn_preprocess(img, is_night):
    x_img = np.expand_dims(img.astype("float32"), axis=0)
    x_isn = np.array([[is_night]], dtype="float32")
    return x_img, x_isn


def cnn_postprocess(x_adv_img):
    return x_adv_img[0]


def rnn_preprocess(img, is_night):
    x_seq = img.reshape(1, IMG_SIZE, IMG_SIZE * 3).astype("float32")
    x_isn = np.array([[is_night]], dtype="float32")
    return x_seq, x_isn


def rnn_postprocess(x_adv_seq):
    return x_adv_seq[0].reshape(IMG_SIZE, IMG_SIZE, 3)


# =========================
# Two-input wrappers for CNN/RNN
# =========================
class TwoInputCNNModel(tf.keras.Model):
    def __init__(self, base_model, fixed_isnight=0.0):
        super().__init__()
        self.base_model = base_model
        self.fixed_isnight = fixed_isnight

    def call(self, x, training=False):
        batch = tf.shape(x)[0]
        isn = tf.ones((batch, 1), dtype=tf.float32) * tf.cast(self.fixed_isnight, tf.float32)
        return self.base_model([x, isn], training=training)


class TwoInputRNNModel(tf.keras.Model):
    def __init__(self, base_model, fixed_isnight=0.0):
        super().__init__()
        self.base_model = base_model
        self.fixed_isnight = fixed_isnight

    def call(self, x, training=False):
        batch = tf.shape(x)[0]
        isn = tf.ones((batch, 1), dtype=tf.float32) * tf.cast(self.fixed_isnight, tf.float32)
        return self.base_model([x, isn], training=training)


# =========================
# ART wrappers
# =========================
class ANNWrapper:
    def __init__(self, model):
        self.model = model
        self.classifier = TensorFlowV2Classifier(
            model=self.model,
            nb_classes=NUM_CLASSES,
            input_shape=(IMG_SIZE * IMG_SIZE * 3 + 1,),
            loss_object=tf.keras.losses.CategoricalCrossentropy(),
            clip_values=(0.0, 1.0),
        )

    def generate(self, attack_name, img, label, is_night):
        x = ann_preprocess(img, is_night)
        y = one_hot(np.array([label]), NUM_CLASSES)

        if attack_name == "fgsm":
            attack = FastGradientMethod(estimator=self.classifier, eps=FGSM_EPS)
        elif attack_name == "pgd":
            attack = ProjectedGradientDescent(
                estimator=self.classifier,
                eps=PGD_EPS,
                eps_step=PGD_EPS_STEP,
                max_iter=PGD_MAX_ITER,
            )
        else:
            raise ValueError(f"Unknown attack: {attack_name}")

        x_adv = attack.generate(x=x, y=y)
        adv_img = ann_postprocess(x_adv)
        return np.clip(adv_img, 0.0, 1.0)


class CNNArtWrapper:
    def __init__(self, model, fixed_isnight=0.0):
        wrapped = TwoInputCNNModel(model, fixed_isnight=fixed_isnight)
        self.classifier = TensorFlowV2Classifier(
            model=wrapped,
            nb_classes=NUM_CLASSES,
            input_shape=(IMG_SIZE, IMG_SIZE, 3),
            loss_object=tf.keras.losses.CategoricalCrossentropy(),
            clip_values=(0.0, 1.0),
        )

    def generate(self, attack_name, img, label, is_night):
        x_img, _ = cnn_preprocess(img, is_night)
        y = one_hot(np.array([label]), NUM_CLASSES)

        if attack_name == "fgsm":
            attack = FastGradientMethod(estimator=self.classifier, eps=FGSM_EPS)
        elif attack_name == "pgd":
            attack = ProjectedGradientDescent(
                estimator=self.classifier,
                eps=PGD_EPS,
                eps_step=PGD_EPS_STEP,
                max_iter=PGD_MAX_ITER,
            )
        else:
            raise ValueError(f"Unknown attack: {attack_name}")

        x_adv = attack.generate(x=x_img, y=y)
        return np.clip(cnn_postprocess(x_adv), 0.0, 1.0)


class RNNArtWrapper:
    def __init__(self, model, fixed_isnight=0.0):
        wrapped = TwoInputRNNModel(model, fixed_isnight=fixed_isnight)
        self.classifier = TensorFlowV2Classifier(
            model=wrapped,
            nb_classes=NUM_CLASSES,
            input_shape=(IMG_SIZE, IMG_SIZE * 3),
            loss_object=tf.keras.losses.CategoricalCrossentropy(),
            clip_values=(0.0, 1.0),
        )

    def generate(self, attack_name, img, label, is_night):
        x_seq, _ = rnn_preprocess(img, is_night)
        y = one_hot(np.array([label]), NUM_CLASSES)

        if attack_name == "fgsm":
            attack = FastGradientMethod(estimator=self.classifier, eps=FGSM_EPS)
        elif attack_name == "pgd":
            attack = ProjectedGradientDescent(
                estimator=self.classifier,
                eps=PGD_EPS,
                eps_step=PGD_EPS_STEP,
                max_iter=PGD_MAX_ITER,
            )
        else:
            raise ValueError(f"Unknown attack: {attack_name}")

        x_adv = attack.generate(x=x_seq, y=y)
        return np.clip(rnn_postprocess(x_adv), 0.0, 1.0)


# =========================
# Main
# =========================
def main():
    df = load_annotations()
    df = df[df["split"].isin(TARGET_SPLITS)].copy()

    ann_model = load_model(MODELS_DIR / "malevis_ann_model.keras")
    cnn_model = load_model(MODELS_DIR / "malevis_cnn_model.keras")
    rnn_model = load_model(MODELS_DIR / "malevis_rnn_model.keras")

    wrappers = {
        "ANN": ANNWrapper(ann_model),
        "CNN": CNNArtWrapper(cnn_model, fixed_isnight=0.0),
        "RNN": RNNArtWrapper(rnn_model, fixed_isnight=0.0),
    }

    log_rows = []

    for model_name, attacks in ATTACK_PLAN.items():
        wrapper = wrappers[model_name]

        for attack_name in attacks:
            out_root = ADV_ROOT / f"{model_name}_{attack_name}"

            print(f"\n=== Generating {model_name}_{attack_name} ===")

            for _, row in tqdm(df.iterrows(), total=len(df), desc=f"{model_name}_{attack_name}"):
                img_path = Path(row["image_id"])
                label = int(row["label"])
                is_night = int(row["isNight"])
                split = str(row["split"])
                class_name = str(row["class_name"]) if "class_name" in row else f"class_{label+1}"

                img = load_image(img_path)
                adv_img = wrapper.generate(attack_name, img, label, is_night)

                rel_name = img_path.name
                out_path = out_root / split / class_name / rel_name
                save_adv_png(adv_img, out_path)

                log_rows.append({
                    "source_model": model_name,
                    "attack": attack_name,
                    "split": split,
                    "class_name": class_name,
                    "orig_path": str(img_path),
                    "adv_path": str(out_path),
                    "label_zero_based": label,
                    "label_csv_1based": label + 1,
                })

    log_df = pd.DataFrame(log_rows)
    log_path = LOGS_DIR / "generated_attacks_log.csv"
    log_df.to_csv(log_path, index=False)

    print(f"\nSaved attack log to: {log_path}")
    print("\nCounts by source_model / attack / split:")
    print(log_df.groupby(["source_model", "attack", "split"]).size())


if __name__ == "__main__":
    main()
