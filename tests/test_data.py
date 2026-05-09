import pandas as pd

from pawpularity_project.data import load_csv, save_csv

import tensorflow as tf
import numpy as np

def test_save_and_load_csv_roundtrip(tmp_path):
    images = np.random.rand(100, 28, 28).astype(np.float32)
    labels = np.random.randint(0, 10, size=(100,))
    dataset = tf.data.Dataset.from_tensor_slices((images, labels))
    dataset = dataset.batch(32)

    out_path = tmp_path / "sample.csv"
    save_csv(dataset, out_path)

    loaded = load_csv(out_path)

    pd.testing.assert_frame_equal(loaded, dataset)
