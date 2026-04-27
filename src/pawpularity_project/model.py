from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.model_selection import train_test_split
from .features import split_features_label


IMG_HEIGHT = 128
IMG_WIDTH = 128
IMG_CHANNELS = 3


def build_model(
    img_height: int = IMG_HEIGHT,
    img_width: int = IMG_WIDTH,
    img_channels: int = IMG_CHANNELS,
) -> tf.keras.Model:
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(img_height, img_width, img_channels)),

        tf.keras.layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
        tf.keras.layers.MaxPooling2D((2, 2)),

        tf.keras.layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
        tf.keras.layers.MaxPooling2D((2, 2)),

        tf.keras.layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
        tf.keras.layers.MaxPooling2D((2, 2)),

        tf.keras.layers.GlobalAveragePooling2D(),

        tf.keras.layers.Dense(1, activation=None),
    ])

    model.compile(
        optimizer="adam",
        loss=tf.keras.losses.MeanSquaredError(),
        metrics=[tf.keras.metrics.RootMeanSquaredError()],
    )

    return model


def train_eval_save(
    X: np.ndarray,
    y: np.ndarray,
    model_path: str,
    random_state: int = 42,
    test_size: float = 0.2,
    epochs: int = 10,
    batch_size: int = 32,
) -> dict[str, float]:
    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
    )

    model = build_model()

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
    )

    val_loss, val_rmse = model.evaluate(X_val, y_val, verbose=0)

    Path(model_path).parent.mkdir(parents=True, exist_ok=True)
    model.save(model_path)

    return {
        "val_loss": float(val_loss),
        "val_rmse": float(val_rmse),
    }


def load_model(path: str) -> tf.keras.Model:
    return tf.keras.models.load_model(path)
