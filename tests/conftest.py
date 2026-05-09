import pandas as pd
import pytest


@pytest.fixture
def sample_ctr_df():
    """Small synthetic dataset for testing."""
    images = np.random.rand(100, 28, 28).astype(np.float32)
    labels = np.random.randint(0, 10, size=(100,))
    dataset = tf.data.Dataset.from_tensor_slices((images, labels))
    dataset = dataset.batch(32)
    return dataset
