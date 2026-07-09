"""
utilities.py
================================================================================
Custom helper module for the Brain MRI Tumor Detection & Localization project.

Everything in this file supports the second (segmentation) stage of the
pipeline, which cannot be expressed with off-the-shelf Keras utilities:

    DataGenerator   Streams (MRI image, tumor mask) pairs to ``model.fit`` in
                    batches so we never hold the full 256x256x3 dataset in RAM.
    tversky         Tversky index - a tunable generalisation of the Dice score
                    used as the segmentation *metric*.
    tversky_loss    ``1 - tversky`` ; the plain differentiable loss.
    focal_tversky   Focal variant of the Tversky loss - the actual training
                    objective, which focuses learning on the small, hard-to-
                    segment tumor regions.
    prediction      Runs the full two-stage inference (classifier -> segmenter)
                    over a test dataframe and returns per-image results.

The Focal Tversky formulation follows Abraham & Khan, "A Novel Focal Tversky
Loss Function with Improved Attention U-Net for Lesion Segmentation" (2018).
"""

import numpy as np
import cv2
from skimage import io
from tensorflow.keras.utils import Sequence
from tensorflow.keras import backend as K


# -----------------------------------------------------------------------------
# Data pipeline
# -----------------------------------------------------------------------------
class DataGenerator(Sequence):
    """Keras ``Sequence`` that yields batches of (image, mask) pairs.

    The LGG dataset has ~1,400 tumour-positive scans; decoding, resizing and
    normalising every 256x256x3 image up front would be wasteful, so we do it
    lazily one batch at a time. Images are scaled to ``[0, 1]`` and the binary
    masks (originally 0 / 255) are collapsed to a single ``{0, 1}`` channel.

    Parameters
    ----------
    ids : list[str]
        Paths to the MRI scans.
    mask : list[str]
        Paths to the matching segmentation masks (same order as ``ids``).
    image_dir : str
        Prefix prepended to every path (defaults to the current directory).
    batch_size : int
        Number of samples per batch.
    img_h, img_w : int
        Target spatial size the network expects.
    shuffle : bool
        Reshuffle sample order at the end of every epoch.
    """

    def __init__(self, ids, mask, image_dir="./", batch_size=16,
                 img_h=256, img_w=256, shuffle=True):
        self.ids = ids
        self.mask = mask
        self.image_dir = image_dir
        self.batch_size = batch_size
        self.img_h = img_h
        self.img_w = img_w
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        """Number of complete batches the generator produces per epoch."""
        return int(np.floor(len(self.ids) / self.batch_size))

    def __getitem__(self, index):
        """Return the ``index``-th batch as ``(X, y)`` arrays."""
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        ids_batch = [self.ids[i] for i in indexes]
        mask_batch = [self.mask[i] for i in indexes]
        return self.__data_generation(ids_batch, mask_batch)

    def on_epoch_end(self):
        """Rebuild (and optionally shuffle) the index order each epoch."""
        self.indexes = np.arange(len(self.ids))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation(self, ids_batch, mask_batch):
        """Read, resize and normalise a single batch of images and masks."""
        X = np.empty((self.batch_size, self.img_h, self.img_w, 3))
        y = np.empty((self.batch_size, self.img_h, self.img_w, 1))

        for i, (img_path, mask_path) in enumerate(zip(ids_batch, mask_batch)):
            # --- MRI scan: resize to the network input and scale to [0, 1] ---
            img = io.imread(img_path)
            img = cv2.resize(img, (self.img_h, self.img_w))
            img = np.array(img, dtype=np.float64) / 255.0

            # --- Mask: resize, take one channel, binarise to {0, 1} ----------
            mask = io.imread(mask_path)
            mask = cv2.resize(mask, (self.img_h, self.img_w))
            mask = np.array(mask, dtype=np.float64) / 255.0
            if mask.ndim == 3:                      # (H, W, 3) -> (H, W)
                mask = mask[:, :, 0]
            mask = np.expand_dims(mask, axis=-1)    # (H, W) -> (H, W, 1)

            X[i, ] = img
            y[i, ] = mask

        return X, y


# -----------------------------------------------------------------------------
# Segmentation loss / metric (Focal Tversky)
# -----------------------------------------------------------------------------
def tversky(y_true, y_pred, smooth=1e-6, alpha=0.7):
    """Tversky index between a ground-truth and predicted mask.

    The Tversky index generalises the Dice coefficient by letting us weight
    false negatives (``alpha``) differently from false positives
    (``1 - alpha``). Tumour pixels are a tiny fraction of every scan, so we set
    ``alpha = 0.7`` to penalise *missed* tumour pixels more heavily than
    spurious ones - exactly the trade-off a medical screening tool wants.
    """
    y_true = K.flatten(y_true)
    y_pred = K.flatten(y_pred)
    true_pos = K.sum(y_true * y_pred)
    false_neg = K.sum(y_true * (1 - y_pred))
    false_pos = K.sum((1 - y_true) * y_pred)
    return (true_pos + smooth) / (
        true_pos + alpha * false_neg + (1 - alpha) * false_pos + smooth
    )


def tversky_loss(y_true, y_pred):
    """Plain Tversky loss: ``1 - tversky`` (perfect overlap -> 0)."""
    return 1 - tversky(y_true, y_pred)


def focal_tversky(y_true, y_pred, gamma=0.75):
    """Focal Tversky loss - the ResUNet's training objective.

    Raising the Tversky loss to a power ``gamma < 1`` reshapes the gradient so
    that easy, already well-segmented pixels contribute less and the network
    keeps focusing on the hard, mis-segmented boundary of the tumour. This is
    what lets the model learn precise masks despite the severe class imbalance.
    """
    return K.pow((1 - tversky(y_true, y_pred)), gamma)


# -----------------------------------------------------------------------------
# End-to-end (classify -> segment) inference
# -----------------------------------------------------------------------------
def prediction(test, model, model_seg):
    """Run the full two-stage pipeline over every scan in ``test``.

    For each MRI scan:
        1. The ResNet50 ``model`` classifies whether a tumour is present.
        2. Only when a tumour is detected does the ResUNet ``model_seg``
           predict a pixel-level mask; empty predictions are discarded.

    Parameters
    ----------
    test : pandas.DataFrame
        Must contain an ``image_path`` column.
    model : tf.keras.Model
        Trained ResNet50 classifier (softmax over [no-tumour, tumour]).
    model_seg : tf.keras.Model
        Trained ResUNet segmentation model.

    Returns
    -------
    image_id : list[str]
        Path of every processed scan.
    mask : list
        Predicted mask array for tumour scans, or the string ``"No mask"``
        when no tumour was found.
    has_mask : list[int]
        ``1`` if a tumour mask was produced, otherwise ``0``. Parallel to the
        two lists above so the caller can merge them back onto ``test``.
    """
    mask, image_id, has_mask = [], [], []

    for path in test.image_path:
        img = io.imread(path)
        img = cv2.resize(img, (256, 256))
        img = np.array(img, dtype=np.float64)
        img_ = np.reshape(img, (1, 256, 256, 3)) / 255.0

        # --- Stage 1: is there a tumour at all? ------------------------------
        is_defect = model.predict(img_)
        if np.argmax(is_defect) == 0:
            image_id.append(path)
            has_mask.append(0)
            mask.append("No mask")
            continue

        # --- Stage 2: localise the tumour with the ResUNet -------------------
        X = np.empty((1, 256, 256, 3))
        img = io.imread(path)
        img = cv2.resize(img, (256, 256))
        X[0, ] = np.array(img, dtype=np.float64) / 255.0
        predict = model_seg.predict(X)

        # Discard predictions that come back empty after rounding.
        if predict.round().astype(int).sum() == 0:
            image_id.append(path)
            has_mask.append(0)
            mask.append("No mask")
        else:
            image_id.append(path)
            has_mask.append(1)
            mask.append(predict)

    return image_id, mask, has_mask
