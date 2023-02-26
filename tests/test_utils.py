import os
from pathlib import Path
import shutil
from typing import List

import numpy as np
import tensorflow as tf


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
    data = generate_sin_wav(frequency, data_length_seconds, samples_per_second=sampling_rate, bits_per_sample=bits_per_sample).tobytes()
    # data = np.array([frequency] * (sampling_rate * data_length_seconds)).astype(np.float16).tobytes()

    riff_chunk = riff_chunk_tag + riff_chunk_size
    wave_chunk = wave_chunk_tag
    fmt_chunk = fmt_chunk_tag + fmt_chunk_size + fmt_chunk_compression + fmt_chunk_channels + fmt_chunk_sample_rate + fmt_chunk_bytes_per_second + fmt_chunk_block_align + fmt_chunk_bits_per_sample
    data_chunk = data_chunk_tag + data_chunk_size + data

    return riff_chunk + wave_chunk + fmt_chunk + data_chunk


def generate_wav_files(n: int, base_path: str) -> List[str]:
    files = []

    os.makedirs(f"{base_path}/test_data/in", exist_ok=False)
    os.makedirs(f"{base_path}/test_data/out", exist_ok=False)

    for i in range(n):
        filename = f"{base_path}/test_data/in/test{i}.wav"
        freq = 440 * (i + 1)
        # freq = 0.0
        with open(filename, "wb") as file:
            data = generate_wav_data(10, freq)
            file.write(data)
        files.append(filename)

        filename = f"{base_path}/test_data/out/test{i}.wav"
        with open(filename, "wb") as file:
            data = generate_wav_data(10, freq)
            file.write(data)
        files.append(filename)

    with open(f"{base_path}/test_data/test.txt", "w") as f:
        for i in range(n):
            f.write(f"test{i}.wav,test{i}.wav\n")
        files.append(f"{base_path}/test_data/test.txt")

    return files


def load_and_parse_tfrecords(base_path: str, out_feature):
    feature_description = {
        'a_in': tf.io.FixedLenFeature([], tf.string),
        'a_out': out_feature
    }

    def _parser(x):
        return tf.io.parse_single_example(x, feature_description)

    raw_ds = tf.data.TFRecordDataset(list(Path(base_path).glob("*.tfrecord")))
    parsed_ds = raw_ds.map(_parser)

    return parsed_ds
