# Brain Tumor Detection & Localization — Technical Report

*A layered deep-learning pipeline for classifying and segmenting brain tumors
in MRI scans.*

---

## 1. Problem & motivation

Reading brain MRI scans is slow, expensive, and depends heavily on radiologist
availability. Two clinically distinct questions have to be answered for every
scan:

1. **Detection** — *is* there a tumor?
2. **Localization** — *where* exactly is it, down to the pixel?

A single classifier answers only the first. A single segmentation network can
answer the second but wastes compute on the ~65% of scans that are healthy and
is harder to train reliably on imbalanced data. This project therefore uses a
**two-stage cascade**: a fast classifier screens every scan, and only the
scans it flags as positive are passed to a heavier segmentation model.

## 2. Dataset

The project uses the **LGG (Lower-Grade Glioma) MRI Segmentation** dataset
originally published on Kaggle
([mateuszbuda/lgg-mri-segmentation](https://www.kaggle.com/mateuszbuda/lgg-mri-segmentation)).
It contains FLAIR MRI scans from TCGA patients, each paired with a
hand-annotated binary mask marking the tumor region.

| Property | Value |
| --- | --- |
| Total scans | 3,929 |
| Tumor-negative (`mask = 0`) | 2,556 |
| Tumor-positive (`mask = 1`) | 1,373 |
| Image size | 256 × 256 × 3 |
| Mask values | 0 (background) / 255 (tumor) |

Two metadata files, `data.csv` and `data_mask.csv`, map every scan to its mask
and store the `patient_id` and the binary `mask` flag.

## 3. Approach

### Stage 1 — ResNet50 classifier (detection)

- **Backbone:** ResNet50 pre-trained on ImageNet, `include_top=False`.
- **Transfer learning:** the convolutional backbone is frozen and only a small
  custom head is trained:
  `AveragePooling2D → Flatten → Dense(256, relu) → Dropout(0.3) →
   Dense(256, relu) → Dropout(0.3) → Dense(2, softmax)`.
- **Why ResNet:** residual "skip connections" let very deep networks train
  without the vanishing-gradient problem, which is why a 50-layer backbone
  transfers so well to a small medical dataset.
- **Training:** `categorical_crossentropy`, Adam, `EarlyStopping` on validation
  loss, `ModelCheckpoint` to keep the best weights.

### Stage 2 — ResUNet segmentation (localization)

Only tumor-positive scans (1,373 images) reach this stage.

- **Architecture:** a U-Net-style encoder–decoder in which every block is a
  **residual block** (`resblock`) rather than a plain convolution stack. The
  encoder downsamples through 5 stages (16 → 32 → 64 → 128 → 256 filters); the
  decoder upsamples symmetrically, and **skip connections** (`upsample_concat`)
  splice encoder feature maps into the decoder so fine spatial detail survives.
- **Output:** a `1 × 1` convolution with a sigmoid activation produces a
  per-pixel tumor probability map at the full 256 × 256 resolution.
- **Custom pipeline:** the encoder/decoder, the batched
  [`DataGenerator`](../utilities.py), and the loss functions live in
  [`utilities.py`](../utilities.py).

### The loss function: Focal Tversky

Tumor pixels are a tiny fraction of each scan, so a naïve Dice/cross-entropy
loss lets the model "win" by predicting mostly background. We instead minimise
the **Focal Tversky loss** (Abraham & Khan, 2018):

- **Tversky index** generalises Dice and lets us weight *false negatives*
  (α = 0.7) more heavily than false positives — missing a tumor is worse than a
  false alarm.
- The **focal** term (raising the loss to γ = 0.75) keeps the gradient focused
  on the hard, mis-segmented boundary of the tumor instead of the easy
  background.

## 4. Results

The classifier was evaluated on a held-out test set of 576 scans.

| Metric | Class 0 (no tumor) | Class 1 (tumor) |
| --- | --- | --- |
| Precision | 0.98 | 0.98 |
| Recall | 0.99 | 0.96 |
| F1-score | 0.99 | 0.97 |

- **Overall classification accuracy: 98.1%.**
- The ResUNet produces tight, well-localized masks.

![Sample segmentation results](../assets/segmentation_results.png)

*For each scan (left to right): MRI · ground-truth mask · AI-predicted mask ·
ground-truth overlay (red) · AI-predicted overlay (green). Full results are
rendered at the end of the notebook.*

## 5. Limitations & future work

- **Single dataset / modality.** Trained only on LGG FLAIR scans; generalization
  to other scanners, sequences, or tumor grades is unverified.
- **No cross-validation of the reported numbers.** Metrics come from a single
  train/test split; k-fold would give tighter confidence intervals.
- **Not a medical device.** This is a research/portfolio project and must not
  be used for real diagnosis.
- **Next steps:** hyperparameter tuning, test-time augmentation, uncertainty
  estimation, and a small Streamlit/Flask demo for interactive inference.

## References

- K. He et al., *Deep Residual Learning for Image Recognition*, 2015.
  <https://arxiv.org/abs/1512.03385>
- O. Ronneberger et al., *U-Net: Convolutional Networks for Biomedical Image
  Segmentation*, 2015. <https://arxiv.org/abs/1505.04597>
- N. Abraham & N. M. Khan, *A Novel Focal Tversky Loss Function with Improved
  Attention U-Net for Lesion Segmentation*, 2018.
  <https://arxiv.org/abs/1810.07842>
- LGG MRI Segmentation dataset:
  <https://www.kaggle.com/mateuszbuda/lgg-mri-segmentation>
