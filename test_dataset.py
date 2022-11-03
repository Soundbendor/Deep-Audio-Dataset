from hashlib import md5
import os
from pathlib import Path
import shutil

import numpy as np
import pytest

from deep_audio_dataset import AudioDataset


def generate_sin_wav(frequency: int, total_seconds: int, samples_per_second: int = 44100):
    values = np.linspace(0.0, total_seconds, samples_per_second * total_seconds)
    values = values * 2 * np.pi * frequency
    values = np.sin(values)
    return values.astype(np.float16)


def generate_wav_data(data_length_seconds: int, frequency: int = 440):
    data_length_samples = data_length_seconds * 44100

    riff_chunk_tag = bytes.fromhex("52 49 46 46")
    riff_chunk_size = (data_length_samples * 2 + 36).to_bytes(4, "little")

    wave_chunk_tag = bytes.fromhex("57 41 56 45")
    # wave_chunk_size = (data_length_samples * 2 + 32).to_bytes(4, "little")

    fmt_chunk_tag = bytes.fromhex("66 6D 74 20")
    fmt_chunk_size = bytes.fromhex("10 00 00 00")
    fmt_chunk_compression = bytes.fromhex("01 00")  # pcm/uncompressed
    fmt_chunk_channels = bytes.fromhex("01 00")
    fmt_chunk_sample_rate = bytes.fromhex("44 AC 00 00")  # 44.1kHz
    fmt_chunk_bytes_per_second = bytes.fromhex("88 58 01 00")
    fmt_chunk_block_align = bytes.fromhex("02 00")
    fmt_chunk_bits_per_sample = bytes.fromhex("10 00")  # 16 bits/sample
    
    data_chunk_tag = bytes.fromhex("64 61 74 61")  # DATA
    data_chunk_size = (data_length_samples * 2).to_bytes(4, "little")
    data = generate_sin_wav(frequency, data_length_seconds).tobytes()

    riff_chunk = riff_chunk_tag + riff_chunk_size
    wave_chunk = wave_chunk_tag
    fmt_chunk = fmt_chunk_tag + fmt_chunk_size + fmt_chunk_compression + fmt_chunk_channels + fmt_chunk_sample_rate + fmt_chunk_bytes_per_second + fmt_chunk_block_align + fmt_chunk_bits_per_sample
    data_chunk = data_chunk_tag + data_chunk_size + data

    return riff_chunk + wave_chunk + fmt_chunk + data_chunk


@pytest.fixture
def wav_files():
    # FIXME should use pytests built in tmp directory fixture
    files = []

    os.makedirs("test_data/in", exist_ok=True)
    os.makedirs("test_data/out", exist_ok=True)

    for i in range(2):
        filename = f"test_data/in/test{i}.wav"
        with open(filename, "wb") as file:
            data = generate_wav_data(10)
            file.write(data)
        files.append(filename)

        filename = f"test_data/out/test{i}.wav"
        with open(filename, "wb") as file:
            data = generate_wav_data(10)
            file.write(data)
        files.append(filename)

    
    with open("test_data/test.txt", "w") as f:
        # this is not as intended (the leading comma)
        f.writelines(["test0.wav,", "test1.wav,"])
        files.append("test_data/test.txt")

    yield files

    for file in files:
        os.remove(file)

    shutil.rmtree("test_data")


def test_audio_dataset_generate(wav_files):
    from deep_audio_dataset import AudioDataset
    ad = AudioDataset("test_data", "test.txt")
    ad.generate()

    assert(Path("test_data/test.txt0.tfrecord").exists())

    with open("test_data/test.txt0.tfrecord", "rb") as f:
        checksum = md5(f.read()).hexdigest()

    possible_checksums = {"d19a0358fa7d1d3ab144857ed91e883d", "c13447cdfb5865c8f95ef44e1bd39ab5"}

    assert(checksum in possible_checksums)
