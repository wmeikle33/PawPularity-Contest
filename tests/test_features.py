import numpy as np
import tensorflow as tf

from pawpularity_project.features import read_and_decode, decode_csv, make_dataset


def test_read_and_decode_returns_resized_image(tmp_path):
    image_path = tmp_path / "dog.jpg"

    image = np.zeros((32, 32, 3), dtype=np.uint8)
    encoded = tf.io.encode_jpeg(image).numpy()
    image_path.write_bytes(encoded)

    result = read_and_decode(
        str(image_path),
        image_size=[224, 224],
        img_channels=3,
    )

    assert result.shape == (224, 224, 3)
    assert result.dtype == tf.float32


def test_decode_csv_returns_image_and_label(tmp_path):
    image_path = tmp_path / "dog.jpg"

    image = np.zeros((32, 32, 3), dtype=np.uint8)
    encoded = tf.io.encode_jpeg(image).numpy()
    image_path.write_bytes(encoded)

    row = tf.constant(f"{image_path},42.0")

    image_tensor, label = decode_csv(
        row,
        img_height=224,
        img_width=224,
        img_channels=3,
    )

    assert image_tensor.shape == (224, 224, 3)
    assert image_tensor.dtype == tf.float32
    assert float(label.numpy()) == 42.0


def test_make_dataset_batches_rows(tmp_path):
    image_path_1 = tmp_path / "dog1.jpg"
    image_path_2 = tmp_path / "dog2.jpg"
    csv_path = tmp_path / "train.csv"

    image = np.zeros((32, 32, 3), dtype=np.uint8)
    encoded = tf.io.encode_jpeg(image).numpy()

    image_path_1.write_bytes(encoded)
    image_path_2.write_bytes(encoded)

    csv_path.write_text(
        f"{image_path_1},10.0\n"
        f"{image_path_2},20.0\n"
    )

    ds = make_dataset(
        csv_path=str(csv_path),
        batch_size=2,
        img_height=224,
        img_width=224,
        img_channels=3,
        shuffle=False,
    )

    images, labels = next(iter(ds))

    assert images.shape == (2, 224, 224, 3)
    assert labels.shape == (2,)
    assert labels.numpy().tolist() == [10.0, 20.0]
