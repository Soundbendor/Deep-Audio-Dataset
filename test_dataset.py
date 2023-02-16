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


def generate_sin_wav(frequency: int, total_seconds: int, samples_per_second: int = 44100, bits_per_sample: int = 16):
    values = np.linspace(0.0, total_seconds, samples_per_second * total_seconds)
    values = values * 2 * np.pi * frequency
    values = np.sin(values) * 32766

    if bits_per_sample == 16:
        bps_dtype = np.float16
    elif bits_per_sample == 32:
        bps_dtype = np.float32
    elif bits_per_sample == 64:
        bps_dtype = np.float64
    else:
        raise ValueError(f"Invalid bits_per_sample used in test: {bits_per_sample}")

    return values.astype(bps_dtype)


def generate_wav_data(
    data_length_seconds: int,
    frequency: int = 440,
    sampling_rate: int = 44100,
    bits_per_sample: int = 16,
    num_channels: int = 1):
    # num_channels is only added to the header, multiple channels worth of data are not actually encoded
    bytes_per_sample = int(bits_per_sample / 8)
    data_length_samples = data_length_seconds * sampling_rate

    riff_chunk_tag = bytes.fromhex("52 49 46 46")
    riff_chunk_size = (data_length_samples * bytes_per_sample + 36).to_bytes(4, "little")

    wave_chunk_tag = bytes.fromhex("57 41 56 45")
    # wave_chunk_size = (data_length_samples * 2 + 32).to_bytes(4, "little")

    fmt_chunk_tag = bytes.fromhex("66 6D 74 20")
    fmt_chunk_size = bytes.fromhex("10 00 00 00")
    fmt_chunk_compression = bytes.fromhex("01 00")  # pcm/uncompressed
    fmt_chunk_channels = num_channels.to_bytes(2, "little")
    fmt_chunk_sample_rate = sampling_rate.to_bytes(4, "little")
    fmt_chunk_bytes_per_second = (bytes_per_sample * sampling_rate).to_bytes(4, "little")
    fmt_chunk_block_align = bytes.fromhex("02 00")
    fmt_chunk_bits_per_sample = bits_per_sample.to_bytes(2, "little")

    data_chunk_tag = bytes.fromhex("64 61 74 61")  # DATA
    data_chunk_size = (data_length_samples * bytes_per_sample).to_bytes(4, "little")
    # data = generate_sin_wav(frequency, data_length_seconds, samples_per_second=sampling_rate, bits_per_sample=bits_per_sample).tobytes()
    data = np.array([frequency] * (sampling_rate * data_length_seconds)).astype(np.float16).tobytes()

    riff_chunk = riff_chunk_tag + riff_chunk_size
    wave_chunk = wave_chunk_tag
    fmt_chunk = fmt_chunk_tag + fmt_chunk_size + fmt_chunk_compression + fmt_chunk_channels + fmt_chunk_sample_rate + fmt_chunk_bytes_per_second + fmt_chunk_block_align + fmt_chunk_bits_per_sample
    data_chunk = data_chunk_tag + data_chunk_size + data

    return riff_chunk + wave_chunk + fmt_chunk + data_chunk


@pytest.fixture
def wav_files():
    # FIXME should use pytests built in tmp directory fixture
    files = []

    os.makedirs("test_data/in", exist_ok=False)
    os.makedirs("test_data/out", exist_ok=False)

    for i in range(2):
        filename = f"test_data/in/test{i}.wav"
        # freq = 440 * (i + 1)
        freq = 0.0
        with open(filename, "wb") as file:
            data = generate_wav_data(10, freq)
            file.write(data)
        files.append(filename)

        filename = f"test_data/out/test{i}.wav"
        with open(filename, "wb") as file:
            data = generate_wav_data(10, freq)
            file.write(data)
        files.append(filename)


    with open("test_data/test.txt", "w") as f:
        f.writelines(["test0.wav,test0.wav\n", "test1.wav,test1.wav"])
        files.append("test_data/test.txt")

    yield files

    for file in files:
        os.remove(file)

    shutil.rmtree("test_data")


def test_audio_dataset_generate(wav_files):
    from deep_audio_dataset import AudioDataset
    ad = AudioDataset("test_data", "test.txt")
    ad.generate(ex_per_file=1)

    assert(Path("test_data/test.txt0.tfrecord").exists())
    assert(Path("test_data/test.txt1.tfrecord").exists())

    checksums = set()

    with open("test_data/test.txt0.tfrecord", "rb") as f:
        checksums.add(md5(f.read()).hexdigest())

    with open("test_data/test.txt1.tfrecord", "rb") as f:
        checksums.add(md5(f.read()).hexdigest())

    print(checksums)

    assert len(checksums) in [1, 2]

    if len(checksums) == 1:
        expected_checksums = [{'75102dcae42f1e2de7630693e7318724'}]
    else:
        expected_checksums = [{'75102dcae42f1e2de7630693e7318724', '0300107e8a25e9f92c3e4f845ed3272e'}]

    assert checksums in expected_checksums


def test_audio_dataset_fail_different_lengths():
    from deep_audio_dataset import AudioDataset

    os.makedirs("test_data/in", exist_ok=False)

    with open("test_data/in/test0.wav", "wb") as file:
        data = generate_wav_data(1)
        file.write(data)

    with open("test_data/in/test1.wav", "wb") as file:
        data = generate_wav_data(2)
        file.write(data)

    with open("test_data/test.txt", "w") as f:
        # this is not as intended (the leading comma)
        f.writelines(["test0.wav,", "test1.wav,"])

    ad = AudioDataset("test_data", "test.txt")

    with pytest.raises(ValueError) as e:
        ad.generate()

    assert(str(e.value) == "Multiple lengths detected (seconds): 1.0, 2.0")

    os.remove("test_data/in/test0.wav")
    os.remove("test_data/in/test1.wav")
    os.remove("test_data/test.txt")
    shutil.rmtree("test_data")


def test_audio_dataset_fail_do_not_exist():
    from deep_audio_dataset import AudioDataset

    os.makedirs("test_data/in", exist_ok=True)

    with open("test_data/in/test0.wav", "wb") as file:
        data = generate_wav_data(1)
        file.write(data)

    with open("test_data/test.txt", "w") as f:
        # this is not as intended (the leading comma)
        f.writelines(["test0.wav,", "test1.wav,"])

    ad = AudioDataset("test_data", "test.txt")

    with pytest.raises(ValueError) as e:
        ad.generate()

    assert(str(e.value) == "The following files do not exist: test_data/in/test1.wav")

    os.remove("test_data/in/test0.wav")
    os.remove("test_data/test.txt")
    shutil.rmtree("test_data")


def test_audio_dataset_fail_multiple_sampling_rates():
    from deep_audio_dataset import AudioDataset

    os.makedirs("test_data/in", exist_ok=True)

    with open("test_data/in/test0.wav", "wb") as file:
        data = generate_wav_data(1)
        file.write(data)

    with open("test_data/in/test1.wav", "wb") as file:
        data = generate_wav_data(1, sampling_rate=88200)
        file.write(data)

    with open("test_data/test.txt", "w") as f:
        # this is not as intended (the leading comma)
        f.writelines(["test0.wav,", "test1.wav,"])

    ad = AudioDataset("test_data", "test.txt")

    with pytest.raises(ValueError) as e:
        ad.generate()

    assert(str(e.value) == "Multiple sampling rates detected: 44100, 88200")

    os.remove("test_data/in/test0.wav")
    os.remove("test_data/in/test1.wav")
    os.remove("test_data/test.txt")
    shutil.rmtree("test_data")


def test_audio_dataset_fail_bits_per_sample():
    from deep_audio_dataset import AudioDataset

    os.makedirs("test_data/in", exist_ok=True)

    with open("test_data/in/test0.wav", "wb") as file:
        data = generate_wav_data(1)
        file.write(data)

    with open("test_data/in/test1.wav", "wb") as file:
        data = generate_wav_data(1, bits_per_sample=32)
        file.write(data)

    with open("test_data/test.txt", "w") as f:
        # this is not as intended (the leading comma)
        f.writelines(["test0.wav,", "test1.wav,"])

    ad = AudioDataset("test_data", "test.txt")

    with pytest.raises(ValueError) as e:
        ad.generate()

    assert(str(e.value) == "Multiple bits per sample detected: 16, 32")

    os.remove("test_data/in/test0.wav")
    os.remove("test_data/in/test1.wav")
    os.remove("test_data/test.txt")
    shutil.rmtree("test_data")


def test_audio_dataset_fail_multiple_channels():
    from deep_audio_dataset import AudioDataset

    os.makedirs("test_data/in", exist_ok=True)

    with open("test_data/in/test0.wav", "wb") as file:
        data = generate_wav_data(1)
        file.write(data)

    with open("test_data/in/test1.wav", "wb") as file:
        data = generate_wav_data(1, num_channels=2)
        file.write(data)

    with open("test_data/test.txt", "w") as f:
        # this is not as intended (the leading comma)
        f.writelines(["test0.wav,", "test1.wav,"])

    ad = AudioDataset("test_data", "test.txt")

    with pytest.raises(ValueError) as e:
        ad.generate()

    assert(str(e.value) == "Multiple number of channels detected: 1, 2")

    os.remove("test_data/in/test0.wav")
    os.remove("test_data/in/test1.wav")
    os.remove("test_data/test.txt")
    shutil.rmtree("test_data")


def test_multilabel_classification(tmp_path):
    os.makedirs(f"{tmp_path}/in", exist_ok=False)

    with open(f"{tmp_path}/in/test0.wav", "wb") as file:
        data = generate_wav_data(1)
        file.write(data)

    with open(f"{tmp_path}/test.txt", "w") as f:
        f.writelines(["test0.wav,1"])


    dataset = MultilabelClassificationAudioDataset(tmp_path, "test.txt")
    dataset.generate()

    print(list(Path(tmp_path).glob("*.tfrecord")))

    feature_description = {
        'a_in': tf.io.FixedLenFeature([], tf.string),
        'a_out': tf.io.FixedLenFeature([], tf.float32)
    }

    def _parser(x):
        return tf.io.parse_single_example(x, feature_description)

    raw_ds = tf.data.TFRecordDataset(list(Path(tmp_path).glob("*.tfrecord")))
    parsed_ds = raw_ds.map(_parser)

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

    print(list(Path(tmp_path).glob("*.tfrecord")))

    feature_description = {
        'a_in': tf.io.FixedLenFeature([], tf.string),
        'a_out': tf.io.FixedLenFeature([2], tf.float32)
    }

    def _parser(x):
        return tf.io.parse_single_example(x, feature_description)

    raw_ds = tf.data.TFRecordDataset(list(Path(tmp_path).glob("*.tfrecord")))
    parsed_ds = raw_ds.map(_parser)

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

    feature_description = {
        'a_in': tf.io.FixedLenFeature([], tf.string),
        'a_out': tf.io.FixedLenFeature([dataset.label_length], tf.float32)
    }

    def _parser(x):
        return tf.io.parse_single_example(x, feature_description)

    raw_ds = tf.data.TFRecordDataset(list(Path(tmp_path).glob("*.tfrecord")))
    parsed_ds = raw_ds.map(_parser)

    print(parsed_ds)

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

    feature_description = {
        'a_in': tf.io.FixedLenFeature([], tf.string),
        'a_out': tf.io.FixedLenFeature([dataset.n_], tf.float32)
    }

    def _parser(x):
        return tf.io.parse_single_example(x, feature_description)

    raw_ds = tf.data.TFRecordDataset(list(Path(tmp_path).glob("*.tfrecord")))
    parsed_ds = raw_ds.map(_parser)

    out_result = sorted([x["a_out"].numpy() for x in parsed_ds], key=lambda x: list(x))

    assert all(out_result[0] == [1.0, 0.0])
