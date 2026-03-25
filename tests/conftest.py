import pandas as pd
import pytest


@pytest.fixture
def sample_ctr_df():
    """Small synthetic CTR dataset for testing."""
    return pd.DataFrame(
        {
            "feature_num": [0, 1, 2, 3, 4, 5],
            "feature_cat": ["a", "a", "b", "b", "c", "c"],
            "click": [0, 1, 0, 1, 0, 1],
        }
    )
