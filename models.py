"""
models.py
================================================================================
Model architectures for the Brain MRI tumor project.

These builders are factored out of the notebook so the *exact same* networks can
be constructed programmatically by ``train.py`` (to regenerate weights) or reused
elsewhere:

    build_classifier()  ResNet50 backbone + custom softmax head  (detection)
    build_resunet()     Residual U-Net -> 1-channel sigmoid mask  (segmentation)

``resblock`` and ``upsample_concat`` are the two building blocks of the ResUNet
and mirror the definitions used in ``Brain_Tumor.ipynb``.
"""

from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.layers import (
    Input, AveragePooling2D, Flatten, Dense, Dropout,
    Conv2D, BatchNormalization, Activation, Add, MaxPool2D,
    UpSampling2D, Concatenate,
)
from tensorflow.keras.models import Model


# -----------------------------------------------------------------------------
# Stage 1 - classifier
# -----------------------------------------------------------------------------
def build_classifier(input_shape=(256, 256, 3), weights="imagenet"):
    """ResNet50 transfer-learning classifier (tumor vs. no-tumor).

    The ImageNet-pretrained convolutional backbone is frozen; only the custom
    head that we bolt on top is trained. ``weights=None`` skips the ImageNet
    download (useful for quickly smoke-testing the architecture).
    """
    base = ResNet50(weights=weights, include_top=False,
                    input_tensor=Input(shape=input_shape))
    for layer in base.layers:
        layer.trainable = False

    head = base.output
    head = AveragePooling2D(pool_size=(4, 4))(head)
    head = Flatten(name="flatten")(head)
    head = Dense(256, activation="relu")(head)
    head = Dropout(0.3)(head)
    head = Dense(256, activation="relu")(head)
    head = Dropout(0.3)(head)
    head = Dense(2, activation="softmax")(head)

    return Model(inputs=base.input, outputs=head)


# -----------------------------------------------------------------------------
# Stage 2 - ResUNet segmenter
# -----------------------------------------------------------------------------
def resblock(X, f):
    """Residual block: a two-conv main path added to a 1x1-conv shortcut.

    This is the ResNet identity trick embedded inside the U-Net - it keeps
    gradients flowing through the deep encoder/decoder.
    """
    X_copy = X

    # main path
    X = Conv2D(f, (1, 1), strides=(1, 1), kernel_initializer="he_normal")(X)
    X = BatchNormalization()(X)
    X = Activation("relu")(X)
    X = Conv2D(f, (3, 3), strides=(1, 1), padding="same",
               kernel_initializer="he_normal")(X)
    X = BatchNormalization()(X)

    # shortcut path
    X_copy = Conv2D(f, (1, 1), strides=(1, 1),
                    kernel_initializer="he_normal")(X_copy)
    X_copy = BatchNormalization()(X_copy)

    # merge
    X = Add()([X, X_copy])
    X = Activation("relu")(X)
    return X


def upsample_concat(x, skip):
    """Upsample ``x`` by 2x and concatenate the matching encoder feature map
    (i.e. the skip connection)."""
    x = UpSampling2D((2, 2))(x)
    return Concatenate()([x, skip])


def build_resunet(input_shape=(256, 256, 3)):
    """Residual U-Net that outputs a full-resolution 1-channel tumor mask."""
    X_input = Input(input_shape)

    # ----- Encoder (downsampling) -----
    conv1 = Conv2D(16, 3, activation="relu", padding="same",
                   kernel_initializer="he_normal")(X_input)
    conv1 = BatchNormalization()(conv1)
    conv1 = Conv2D(16, 3, activation="relu", padding="same",
                   kernel_initializer="he_normal")(conv1)
    conv1 = BatchNormalization()(conv1)
    pool1 = MaxPool2D((2, 2))(conv1)

    conv2 = resblock(pool1, 32)
    pool2 = MaxPool2D((2, 2))(conv2)

    conv3 = resblock(pool2, 64)
    pool3 = MaxPool2D((2, 2))(conv3)

    conv4 = resblock(pool3, 128)
    pool4 = MaxPool2D((2, 2))(conv4)

    # ----- Bottleneck -----
    conv5 = resblock(pool4, 256)

    # ----- Decoder (upsampling, with skip connections) -----
    up1 = resblock(upsample_concat(conv5, conv4), 128)
    up2 = resblock(upsample_concat(up1, conv3), 64)
    up3 = resblock(upsample_concat(up2, conv2), 32)
    up4 = resblock(upsample_concat(up3, conv1), 16)

    output = Conv2D(1, (1, 1), padding="same", activation="sigmoid")(up4)
    return Model(inputs=X_input, outputs=output)
