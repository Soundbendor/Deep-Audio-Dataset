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

    ad._ensure_generated_records_directory()
    ds = ad._save_generated_dataset("test", [i for i in range(2)])

    for x, y in ds.as_numpy_iterator():
        assert len(x) == 441000
        assert len(y) == 441000
        assert list(x) == list(y)


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
    dataset._ensure_generated_records_directory()
    actual_results = [(x, y) for x, y in dataset._save_generated_dataset("test", [i for i in range(1)]).as_numpy_iterator()]

    assert len(actual_results) == 1
    assert list(actual_results[0][1]) == [1.0]


def test_multilabel_classification_two_labels(tmp_path):
    os.makedirs(f"{tmp_path}/in", exist_ok=False)

    with open(f"{tmp_path}/in/test0.wav", "wb") as file:
        data = generate_wav_data(1)
        file.write(data)

    with open(f"{tmp_path}/test.txt", "w") as f:
        f.writelines(["test0.wav,01"])


    dataset = MultilabelClassificationAudioDataset(tmp_path, "test.txt")
    dataset._ensure_generated_records_directory()
    actual_results = [(x, y) for x, y in dataset._save_generated_dataset("test", [i for i in range(1)]).as_numpy_iterator()]

    assert len(actual_results) == 1
    assert all(actual_results[0][1] == [0.0, 1.0])


def test_multilabel_classification_two_samples(tmp_path):
    os.makedirs(f"{tmp_path}/in", exist_ok=False)

    with open(f"{tmp_path}/in/test0.wav", "wb") as file:
        data = generate_wav_data(1)
        file.write(data)

    with open(f"{tmp_path}/in/test1.wav", "wb") as file:
        data = generate_wav_data(1)
        file.write(data)

    with open(f"{tmp_path}/test.txt", "w") as f:
        f.write("test0.wav,01\ntest1.wav,10")


    dataset = MultilabelClassificationAudioDataset(tmp_path, "test.txt")
    dataset._ensure_generated_records_directory()
    actual_results = [(x, y) for x, y in dataset._save_generated_dataset("test", [i for i in range(2)]).as_numpy_iterator()]
    actual_results = sorted(actual_results, key=lambda x: x[0][0])

    assert len(actual_results) == 2
    assert all(actual_results[0][1] == [0.0, 1.0])
    assert all(actual_results[1][1] == [1.0, 0.0])


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

    assert dataset.metadata is not None
    assert dataset.metadata["test0.wav"] == {"artist": "Artist0"}
    assert dataset.metadata_stats["fields"] == ["artist"]
    assert dataset.metadata_stats["values"]["artist"] == ["Artist0"]


def test_dataset_kfold_metadata(tmp_path):
    os.makedirs(f"{tmp_path}/in", exist_ok=False)
    os.makedirs(f"{tmp_path}/out", exist_ok=False)

    for i in range(9):
        with open(f"{tmp_path}/in/test{i}.wav", "wb") as file:
            data = generate_wav_data(1)
            file.write(data)

        with open(f"{tmp_path}/out/test{i}.wav", "wb") as file:
            data = generate_wav_data(1)
            file.write(data)

    with open(f"{tmp_path}/test.txt", "w") as f:
        f.writelines([f"test{i}.wav,test{i}.wav\n" for i in range(9)])

    with open(f"{tmp_path}/metadata.txt", "w") as f:
        metadata = {f"test{i}.wav": {"artist": f"Artist{i % 3}"} for i in range(9)}
        json.dump(metadata, f)

    dataset = AudioDataset(tmp_path, "test.txt", metadata_file="metadata.txt")

    meta_values = []

    for train, test, meta_value in dataset.kfold_on_metadata("artist"):
        assert len(list(train.as_numpy_iterator())) == 6
        assert len(list(test.as_numpy_iterator())) == 3
        meta_values.append(meta_value)

    assert len(meta_values) == 3
    assert "Artist0" in meta_values
    assert "Artist1" in meta_values
    assert "Artist2" in meta_values


def test_regression_dataset(tmp_path):
    os.makedirs(f"{tmp_path}/in", exist_ok=False)

    for i in range(1):
        with open(f"{tmp_path}/in/test{i}.wav", "wb") as file:
            data = generate_wav_data(1)
            file.write(data)

    with open(f"{tmp_path}/test.txt", "w") as f:
        f.writelines([f"test{i}.wav,1.0,0.0\n" for i in range(1)])

    dataset = RegressionAudioDataset(tmp_path, "test.txt")
    assert dataset.n_ == 2

    actual_results = [(x, y) for x, y in dataset._save_generated_dataset("test", [i for i in range(1)]).as_numpy_iterator()]

    assert len(actual_results) == 1
    assert all(actual_results[0][1] == [1.0, 0.0])


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
