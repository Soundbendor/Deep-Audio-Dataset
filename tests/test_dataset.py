from hashlib import md5
import json
import os
from pathlib import Path
import shutil

import numpy as np
import pytest
import tensorflow as tf

from deep_audio_dataset import (
    AudioDataset,
    MultilabelClassificationAudioDataset,
    RegressionAudioDataset,
)

from test_utils import generate_wav_files, generate_wav_data, load_and_parse_tfrecords


@pytest.fixture
def two_wav_files(tmp_path):
    return generate_wav_files(2, tmp_path), tmp_path


@pytest.fixture
def ten_wav_files(tmp_path):
    return generate_wav_files(10, tmp_path), tmp_path


def test_audio_dataset_generate(two_wav_files):
    base_path = two_wav_files[1]
    ad = AudioDataset(f"{base_path}/test_data", "test.txt")
    ad.generate(ex_per_file=1)

    assert(Path(f"{base_path}/test_data/test.txt0.tfrecord").exists())
    assert(Path(f"{base_path}/test_data/test.txt1.tfrecord").exists())

    checksums = set()

    with open(f"{base_path}/test_data/test.txt0.tfrecord", "rb") as f:
        checksums.add(md5(f.read()).hexdigest())

    with open(f"{base_path}/test_data/test.txt1.tfrecord", "rb") as f:
        checksums.add(md5(f.read()).hexdigest())

    print(checksums)

    assert len(checksums) in [1, 2]

    if len(checksums) == 1:
        expected_checksums = [{'75102dcae42f1e2de7630693e7318724'}]
    else:
        expected_checksums = [{'75102dcae42f1e2de7630693e7318724', '0300107e8a25e9f92c3e4f845ed3272e'}]

    assert checksums in expected_checksums


def test_audio_dataset_fail_different_lengths(tmp_path):
    for directory in ["in", "out"]:
        os.makedirs(f"{tmp_path}/test_data/{directory}")

        with open(f"{tmp_path}/test_data/{directory}/test0.wav", "wb") as file:
            data = generate_wav_data(1)
            file.write(data)

        with open(f"{tmp_path}/test_data/{directory}/test1.wav", "wb") as file:
            data = generate_wav_data(2)
            file.write(data)

    with open(f"{tmp_path}/test_data/test.txt", "w") as f:
        f.writelines("test0.wav,test0.wav\ntest1.wav,test1.wav")

    with pytest.raises(ValueError) as e:
        AudioDataset(f"{tmp_path}/test_data", "test.txt")

    assert(str(e.value) == "Multiple lengths detected (seconds): 1.0, 2.0")


def test_audio_dataset_fail_do_not_exist(tmp_path):
    from deep_audio_dataset import AudioDataset

    os.makedirs(f"{tmp_path}/test_data/in")
    os.makedirs(f"{tmp_path}/test_data/out")

    with open(f"{tmp_path}/test_data/in/test0.wav", "wb") as file:
        data = generate_wav_data(1)
        file.write(data)

    with open(f"{tmp_path}/test_data/test.txt", "w") as f:
        f.writelines("test0.wav,test1.wav\ntest1.wav,test1.wav")

    with pytest.raises(ValueError) as e:
        AudioDataset(f"{tmp_path}/test_data", "test.txt")

    assert(str(e.value) == f"The following files do not exist: {tmp_path}/test_data/in/test1.wav")


def test_audio_dataset_fail_multiple_sampling_rates(tmp_path):
    for directory in ["in", "out"]:
        os.makedirs(f"{tmp_path}/test_data/{directory}")

        with open(f"{tmp_path}/test_data/{directory}/test0.wav", "wb") as file:
            data = generate_wav_data(1)
            file.write(data)

        with open(f"{tmp_path}/test_data/{directory}/test1.wav", "wb") as file:
            data = generate_wav_data(1, sampling_rate=88200)
            file.write(data)

    with open(f"{tmp_path}/test_data/test.txt", "w") as f:
        f.writelines("test0.wav,test0.wav\ntest1.wav,test1.wav")

    with pytest.raises(ValueError) as e:
        AudioDataset(f"{tmp_path}/test_data", "test.txt")

    assert(str(e.value) == "Multiple sampling rates detected: 44100, 88200")


def test_audio_dataset_fail_bits_per_sample(tmp_path):
    os.makedirs(f"{tmp_path}/test_data/in", exist_ok=True)
    os.makedirs(f"{tmp_path}/test_data/out", exist_ok=True)

    for directory in ["in", "out"]:
        with open(f"{tmp_path}/test_data/{directory}/test0.wav", "wb") as file:
            data = generate_wav_data(1)
            file.write(data)

        with open(f"{tmp_path}/test_data/in/test1.wav", "wb") as file:
            data = generate_wav_data(1, bits_per_sample=32)
            file.write(data)

    with open(f"{tmp_path}/test_data/test.txt", "w") as f:
        f.write("test0.wav,test0.wav\ntest1.wav,test1.wav")

    with pytest.raises(ValueError) as e:
        AudioDataset(f"{tmp_path}/test_data", "test.txt")

    assert(str(e.value) == "Multiple bits per sample detected: 16, 32")


def test_audio_dataset_fail_multiple_channels(tmp_path):
    os.makedirs(f"{tmp_path}/test_data/in")
    os.makedirs(f"{tmp_path}/test_data/out")

    with open(f"{tmp_path}/test_data/in/test0.wav", "wb") as file:
        data = generate_wav_data(1)
        file.write(data)

    with open(f"{tmp_path}/test_data/in/test1.wav", "wb") as file:
        data = generate_wav_data(1, num_channels=2)
        file.write(data)

    with open(f"{tmp_path}/test_data/out/test0.wav", "wb") as file:
        data = generate_wav_data(1)
        file.write(data)

    with open(f"{tmp_path}/test_data/out/test1.wav", "wb") as file:
        data = generate_wav_data(1, num_channels=2)
        file.write(data)

    with open(f"{tmp_path}/test_data/test.txt", "w") as f:
        f.write("test0.wav,test0.wav\ntest1.wav,test1.wav")

    with pytest.raises(ValueError) as e:
        AudioDataset(f"{tmp_path}/test_data", "test.txt")

    assert(str(e.value) == "Multiple number of channels detected: 1, 2")


def test_multilabel_classification(tmp_path):
    os.makedirs(f"{tmp_path}/in", exist_ok=False)

    with open(f"{tmp_path}/in/test0.wav", "wb") as file:
        data = generate_wav_data(1)
        file.write(data)

    with open(f"{tmp_path}/test.txt", "w") as f:
        f.writelines(["test0.wav,1"])


    dataset = MultilabelClassificationAudioDataset(tmp_path, "test.txt")
    dataset.generate()

    parsed_ds = load_and_parse_tfrecords(tmp_path, tf.io.FixedLenFeature([], tf.float32))

    out_result = [x["a_out"].numpy() for x in parsed_ds]

    assert out_result == [1.0]


def test_multilabel_classification_two_labels(tmp_path):
    os.makedirs(f"{tmp_path}/in", exist_ok=False)

    with open(f"{tmp_path}/in/test0.wav", "wb") as file:
        data = generate_wav_data(1)
        file.write(data)

    with open(f"{tmp_path}/test.txt", "w") as f:
        f.writelines(["test0.wav,01"])


    dataset = MultilabelClassificationAudioDataset(tmp_path, "test.txt")
    dataset.generate()

    parsed_ds = load_and_parse_tfrecords(tmp_path, tf.io.FixedLenFeature([2], tf.float32))

    out_result = [x["a_out"].numpy() for x in parsed_ds]

    assert all(out_result[0] == [0.0, 1.0])


def test_multilabel_classification_two_samples(tmp_path):
    os.makedirs(f"{tmp_path}/in", exist_ok=False)

    with open(f"{tmp_path}/in/test0.wav", "wb") as file:
        data = generate_wav_data(1)
        file.write(data)

    with open(f"{tmp_path}/in/test1.wav", "wb") as file:
        data = generate_wav_data(1)
        file.write(data)

    with open(f"{tmp_path}/test.txt", "w") as f:
        f.writelines(["test0.wav,01", "\ntest1.wav,10"])


    dataset = MultilabelClassificationAudioDataset(tmp_path, "test.txt")
    dataset.generate()

    parsed_ds = load_and_parse_tfrecords(tmp_path, tf.io.FixedLenFeature([dataset.label_length], tf.float32))

    out_result = sorted([x["a_out"].numpy() for x in parsed_ds], key=lambda x: list(x))

    assert all(out_result[0] == [0.0, 1.0])
    assert all(out_result[1] == [1.0, 0.0])


def test_dataset_with_metadata(tmp_path):
    os.makedirs(f"{tmp_path}/in", exist_ok=False)
    os.makedirs(f"{tmp_path}/out", exist_ok=False)

    with open(f"{tmp_path}/in/test0.wav", "wb") as file:
        data = generate_wav_data(1)
        file.write(data)

    with open(f"{tmp_path}/out/test0.wav", "wb") as file:
        data = generate_wav_data(1)
        file.write(data)

    with open(f"{tmp_path}/test.txt", "w") as f:
        f.writelines(["test0.wav,test0.wav"])

    with open(f"{tmp_path}/metadata.txt", "w") as f:
        metadata = {
            "test0.wav": {
                "artist": "Artist0"
            }
        }
        json.dump(metadata, f)

    dataset = AudioDataset(tmp_path, "test.txt", metadata_file="metadata.txt")
    dataset.generate()

    assert dataset.metadata is not None
    assert dataset.metadata["test0.wav"] == {"artist": "Artist0"}
    assert dataset.metadata_stats["fields"] == ["artist"]
    assert dataset.metadata_stats["values"]["artist"] == ["Artist0"]

    feature_description = {
        'a_in': tf.io.FixedLenFeature([], tf.string),
        'a_out': tf.io.FixedLenFeature([], tf.string),
        'metadata': tf.io.VarLenFeature(tf.string)
    }

    def _parser(x):
        return tf.io.parse_single_example(x, feature_description)

    raw_ds = tf.data.TFRecordDataset(list(Path(tmp_path).glob("*.tfrecord")))
    parsed_ds = raw_ds.map(_parser)

    recovered_metadata = [json.loads(x["metadata"].values.numpy()[0].decode()) for x in parsed_ds]

    assert recovered_metadata == [metadata["test0.wav"]]


def test_dataset_kfold_metadata(tmp_path):
    os.makedirs(f"{tmp_path}/in", exist_ok=False)
    os.makedirs(f"{tmp_path}/out", exist_ok=False)

    for i in range(10):
        with open(f"{tmp_path}/in/test{i}.wav", "wb") as file:
            data = generate_wav_data(1)
            file.write(data)

        with open(f"{tmp_path}/out/test{i}.wav", "wb") as file:
            data = generate_wav_data(1)
            file.write(data)

    with open(f"{tmp_path}/test.txt", "w") as f:
        f.writelines([f"test{i}.wav,test{i}.wav\n" for i in range(10)])

    with open(f"{tmp_path}/metadata.txt", "w") as f:
        metadata = {f"test{i}.wav": {"artist": f"Artist{i % 2}"} for i in range(10)}
        json.dump(metadata, f)

    dataset = AudioDataset(tmp_path, "test.txt", metadata_file="metadata.txt")
    dataset.generate()

    meta_values = []

    for train, test, meta_value in dataset.kfold_on_metadata("artist"):
        meta_values.append(meta_value)

    assert len(meta_values) == 2
    assert "Artist0" in meta_values
    assert "Artist1" in meta_values


def test_regression_dataset(tmp_path):
    os.makedirs(f"{tmp_path}/in", exist_ok=False)

    for i in range(1):
        with open(f"{tmp_path}/in/test{i}.wav", "wb") as file:
            data = generate_wav_data(1)
            file.write(data)

    with open(f"{tmp_path}/test.txt", "w") as f:
        f.writelines([f"test{i}.wav,1.0,0.0\n" for i in range(1)])

    dataset = RegressionAudioDataset(tmp_path, "test.txt")
    dataset.generate()

    assert dataset.n_ == 2

    parsed_ds = load_and_parse_tfrecords(tmp_path, tf.io.FixedLenFeature([2], tf.float32))

    out_result = sorted([x["a_out"].numpy() for x in parsed_ds], key=lambda x: list(x))

    assert all(out_result[0] == [1.0, 0.0])


def test_train_test_split(ten_wav_files):
    file_list, base_path = ten_wav_files

    dataset = AudioDataset(os.path.join(base_path, "test_data"), file_list[-1].split("/")[-1], "nora")

    train, test = dataset.train_test_split(0.8)

    assert isinstance(train, tf.data.Dataset)
    assert isinstance(test, tf.data.Dataset)

    train_set = {(tuple(x[0]), tuple(x[1])) for x in train.as_numpy_iterator()}
    test_set = {(tuple(x[0]), tuple(x[1])) for x in test.as_numpy_iterator()}

    assert len(train_set) == 8
    assert len(test_set) == 2

    assert len(train_set.intersection(test_set)) == 0

    for x, y in list(train_set):
        assert x == y

    for x, y in list(test_set):
        assert x == y
