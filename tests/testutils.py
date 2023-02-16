import numpy as np


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



