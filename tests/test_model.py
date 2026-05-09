import pandas as pd

from pawpularity_project.model import build_pipeline, train_eval_save, load_model


def make_small_training_df():
  images = np.random.rand(100, 28, 28).astype(np.float32) 
  labels = np.random.randint(0, 10, size=(100,))
  dataset = tf.data.Dataset.from_tensor_slices((images, labels))
  dataset = dataset.batch(32)

def test_build_pipeline_can_fit_and_predict():


def test_train_eval_save_writes_model_and_returns_metrics(tmp_path):

def test_load_model_loads_saved_pipeline(tmp_path):
