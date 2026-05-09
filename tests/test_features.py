import pandas as pd

from pawpularity_project.features import split_features_label, auto_preprocess

def test_split_features_label_splits_target_correctly():
    images = np.random.rand(100, 28, 28).astype(np.float32)
    labels = np.random.randint(0, 10, size=(100,))
    dataset = tf.data.Dataset.from_tensor_slices((images, labels))
    dataset = dataset.batch(32)


def test_auto_preprocess_runs_on_small_mixed_dataframe():

