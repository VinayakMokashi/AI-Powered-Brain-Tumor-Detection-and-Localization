"""
predict.py
================================================================================
Run the trained two-stage pipeline on new MRI scan(s) from the command line.

    # single scan
    python predict.py --image scan.tif

    # a whole folder of scans
    python predict.py --dir ./some_scans --out-dir predictions

For every input the script:
    1. Runs the ResNet50 classifier to decide whether a tumor is present.
    2. If it is, runs the ResUNet to predict the tumor mask and saves a
       side-by-side overlay PNG (predicted region in green) to --out-dir.

It expects the four trained artifacts produced by ``train.py`` (or your own
training run). Paths default to the repo-root filenames but can be overridden:

    resnet-50-MRI.json / weights.hdf5        (classifier)
    ResUNet-MRI.json   / weights_seg.hdf5    (segmenter)
"""

import argparse
import glob
import os

import cv2
import numpy as np
from skimage import io

import matplotlib
matplotlib.use("Agg")  # headless: render figures straight to disk
import matplotlib.pyplot as plt

import tensorflow as tf


def load_models(clf_json, clf_weights, seg_json, seg_weights):
    """Rebuild both networks from their JSON architecture + load their weights.

    Inference does not require compilation, so we skip it (and thus don't need
    the custom Focal Tversky objects here)."""
    with open(clf_json) as f:
        classifier = tf.keras.models.model_from_json(f.read())
    classifier.load_weights(clf_weights)

    with open(seg_json) as f:
        segmenter = tf.keras.models.model_from_json(f.read())
    segmenter.load_weights(seg_weights)

    return classifier, segmenter


def _load_scan(path):
    """Read an MRI scan and resize it to the 256x256x3 the models expect."""
    img = io.imread(path)
    img = cv2.resize(img, (256, 256))
    return np.array(img, dtype=np.float64)


def predict_one(path, classifier, segmenter, out_dir):
    """Classify one scan and, if positive, segment it and save an overlay.

    Returns True if a tumor was detected, else False."""
    raw = _load_scan(path)
    x = (raw / 255.0).reshape(1, 256, 256, 3)

    # --- Stage 1: is there a tumor? -----------------------------------------
    if np.argmax(classifier.predict(x, verbose=0)) == 0:
        print(f"[no tumor]  {path}")
        return False

    # --- Stage 2: where is it? ----------------------------------------------
    mask = segmenter.predict(x, verbose=0)[0].squeeze().round()

    mri_rgb = cv2.cvtColor(raw.astype(np.uint8), cv2.COLOR_BGR2RGB)
    overlay = mri_rgb.copy()
    overlay[mask == 1] = (0, 255, 0)  # predicted tumor region in green

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(
        out_dir, os.path.splitext(os.path.basename(path))[0] + "_prediction.png"
    )

    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    axs[0].imshow(mri_rgb);        axs[0].set_title("MRI")
    axs[1].imshow(mask, cmap="gray"); axs[1].set_title("Predicted mask")
    axs[2].imshow(overlay);        axs[2].set_title("MRI + prediction")
    for ax in axs:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)

    print(f"[TUMOR]     {path}  ->  {out_path}")
    return True


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--image", help="Path to a single MRI scan")
    source.add_argument("--dir", help="Folder of MRI scans (*.tif/*.png/*.jpg)")
    parser.add_argument("--clf-json", default="resnet-50-MRI.json")
    parser.add_argument("--clf-weights", default="weights.hdf5")
    parser.add_argument("--seg-json", default="ResUNet-MRI.json")
    parser.add_argument("--seg-weights", default="weights_seg.hdf5")
    parser.add_argument("--out-dir", default="predictions",
                        help="Where to save overlay PNGs (default: ./predictions)")
    args = parser.parse_args()

    classifier, segmenter = load_models(
        args.clf_json, args.clf_weights, args.seg_json, args.seg_weights
    )

    if args.image:
        paths = [args.image]
    else:
        exts = ("*.tif", "*.tiff", "*.png", "*.jpg", "*.jpeg")
        paths = sorted(f for e in exts for f in glob.glob(os.path.join(args.dir, e)))
        if not paths:
            raise SystemExit(f"No images found in {args.dir}")

    tumors = sum(predict_one(p, classifier, segmenter, args.out_dir) for p in paths)
    print(f"\nProcessed {len(paths)} scan(s); tumor detected in {tumors}.")


if __name__ == "__main__":
    main()
