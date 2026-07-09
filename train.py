"""
train.py
================================================================================
Reproducible, end-to-end training for the Brain MRI tumor pipeline.

Running this regenerates the four files that ``Brain_Tumor.ipynb`` loads in its
evaluation sections, so the notebook becomes runnable top-to-bottom without any
externally-hosted weights:

    resnet-50-MRI.json   +   weights.hdf5        (ResNet50 classifier)
    ResUNet-MRI.json     +   weights_seg.hdf5    (ResUNet segmenter)

Usage
-----
    # train both stages
    python train.py --data-dir /path/to/Brain_MRI

    # only the segmenter, more epochs
    python train.py --data-dir /path/to/Brain_MRI --stage segmenter --seg-epochs 80

``--data-dir`` must contain ``data_mask.csv`` and the per-patient scan folders
referenced by its ``image_path`` / ``mask_path`` columns. A GPU is strongly
recommended.
"""

import argparse
import os

import pandas as pd
import tensorflow as tf
from keras_preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from models import build_classifier, build_resunet
from utilities import DataGenerator, focal_tversky, tversky


def _save_architecture(model, json_path):
    """Serialize a model's architecture to JSON (weights are handled by the
    ModelCheckpoint callback so we always keep the *best*, not the last, epoch)."""
    with open(json_path, "w") as f:
        f.write(model.to_json())
    print(f"  saved architecture -> {json_path}")


def _callbacks(weights_path):
    """Early stopping + best-weights checkpointing shared by both stages."""
    return [
        EarlyStopping(monitor="val_loss", mode="min", verbose=1, patience=20),
        ModelCheckpoint(filepath=weights_path, verbose=1,
                        save_best_only=True, save_weights_only=True),
    ]


def train_classifier(brain_df, out_dir, epochs, batch_size):
    """Train the ResNet50 detector and save resnet-50-MRI.json + weights.hdf5."""
    print("\n=== Stage 1: ResNet50 classifier ===")
    df = brain_df.drop(columns=["patient_id"]).copy()
    df["mask"] = df["mask"].astype(str)  # flow_from_dataframe needs string labels

    train, _test = train_test_split(df, test_size=0.15, random_state=42)

    datagen = ImageDataGenerator(rescale=1 / 255.0, validation_split=0.15)
    common = dict(dataframe=train, directory="./", x_col="image_path",
                  y_col="mask", batch_size=batch_size, shuffle=True,
                  class_mode="categorical", target_size=(256, 256))
    train_gen = datagen.flow_from_dataframe(subset="training", **common)
    val_gen = datagen.flow_from_dataframe(subset="validation", **common)

    model = build_classifier()
    model.compile(loss="categorical_crossentropy", optimizer="adam",
                  metrics=["accuracy"])

    weights_path = os.path.join(out_dir, "weights.hdf5")
    model.fit(train_gen, steps_per_epoch=train_gen.n // batch_size,
              epochs=epochs, validation_data=val_gen,
              validation_steps=val_gen.n // batch_size,
              callbacks=_callbacks(weights_path))

    _save_architecture(model, os.path.join(out_dir, "resnet-50-MRI.json"))
    print(f"  best weights -> {weights_path}")


def train_segmenter(brain_df, out_dir, epochs, batch_size):
    """Train the ResUNet segmenter and save ResUNet-MRI.json + weights_seg.hdf5."""
    print("\n=== Stage 2: ResUNet segmenter ===")
    df = brain_df[brain_df["mask"] == 1]  # only tumor-positive scans
    X_train, X_val = train_test_split(df, test_size=0.15, random_state=42)

    train_gen = DataGenerator(list(X_train.image_path), list(X_train.mask_path),
                              batch_size=batch_size)
    val_gen = DataGenerator(list(X_val.image_path), list(X_val.mask_path),
                            batch_size=batch_size)

    model = build_resunet()
    adam = tf.keras.optimizers.Adam(learning_rate=0.05, epsilon=0.1)
    model.compile(optimizer=adam, loss=focal_tversky, metrics=[tversky])

    weights_path = os.path.join(out_dir, "weights_seg.hdf5")
    model.fit(train_gen, epochs=epochs, validation_data=val_gen,
              callbacks=_callbacks(weights_path))

    _save_architecture(model, os.path.join(out_dir, "ResUNet-MRI.json"))
    print(f"  best weights -> {weights_path}")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--data-dir", default=".",
                        help="Folder holding data_mask.csv and the scan folders")
    parser.add_argument("--output-dir", default=".",
                        help="Where to write the .json / .hdf5 files")
    parser.add_argument("--clf-epochs", type=int, default=30)
    parser.add_argument("--seg-epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--stage", choices=["all", "classifier", "segmenter"],
                        default="all")
    args = parser.parse_args()

    out_dir = os.path.abspath(args.output_dir)
    os.makedirs(out_dir, exist_ok=True)
    # image paths in the CSV are relative to the data dir, so run from there
    os.chdir(args.data_dir)
    brain_df = pd.read_csv("data_mask.csv")
    print(f"Loaded {len(brain_df)} scans "
          f"({int((brain_df['mask'] == 1).sum())} tumor-positive).")

    if args.stage in ("all", "classifier"):
        train_classifier(brain_df, out_dir, args.clf_epochs, args.batch_size)
    if args.stage in ("all", "segmenter"):
        train_segmenter(brain_df, out_dir, args.seg_epochs, args.batch_size)

    print("\nDone. The notebook's evaluation cells can now load these files.")


if __name__ == "__main__":
    main()
