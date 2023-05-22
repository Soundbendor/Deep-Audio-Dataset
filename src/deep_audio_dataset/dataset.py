"""Basic Audio Dataset classes."""
from abc import ABC, abstractmethod
from functools import partial
from itertools import chain
import json
import os
from pathlib import Path
import random
import time
from typing import (
    Any,
    List,
    Mapping,
    Optional,
    Tuple,
    Union,
)
import wave

import numpy as np
import tensorflow as tf


class BaseAudioDataset(ABC):
    """Base class for audio datasets."""

    def __init__(
            self,
            directory: str,
            index_file: str,
            seed: Optional[Any] = None,
            metadata_file: Optional[str] = None,
            generate_tfrecords: bool = True,
            shuffle_size: Optional[int] = None
        ) -> None:
        """Initialize the BaseAudioDataset.

        Args:
            directory (str): Base directory for the dataset to use.
            index_file (str): Name of the data index file.
            seed (any, optional): Seed to use for the random number generator.
            metadata_file (str, optional): Name of the file that contains metadata about the dataset.
                If None, no metadata will be loaded. Defaults to None.
            generate_tfrecords (bool, optional): If True, the tfrecords will be generated in the constructor if
                they don't already exist. If False then tfrecords will be generated only once they are needed.
                Defaults to True.
            shuffle_size (int, optional): Size of the shuffle buffer to use when shuffling the dataset.
                If None, 10% or 1000 elements will be used (whichever is larger).
                If 0, no shuffling will be done.
                Defaults to None.
        """
        if seed is None:
            self._rng = random.Random(time.time())
        else:
            self._rng = random.Random(seed)

        #store args as class members
        self._dir = directory
        self._name = index_file
        self._shuffle_size = shuffle_size

        self._metadata_file = metadata_file
        self.metadata = None
        self.metadata_stats = None

        self.inputs = None
        self.outputs = None
        self.input_len = None
        self.num_examples = None

        self._load_index()
        self._load_metadata()

        if generate_tfrecords:
            self._ensure_tfrecords_exist()

    def all(self) -> tf.data.Dataset:  # noqa: A003
        """Return the entire dataset as a tf.data.Dataset.

        Returns:
            Dataset: The entire dataset.
        """
        raw_ds = self._load_raw_ds()
        return self._generate_filtered_ds(raw_ds, list(range(self.num_examples)))

    def train_test_split(
        self,
        train_size: Optional[float] = None,
        test_size: Optional[float] = None
    ) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
        """Split the dataset into two disjoint subsets: train set and test set.

        While the split does not have to cover the entire dataset,
        the sizes cannot be larger than the whole dataset (i.e. this must be true: train_size + test_size <= 1).
        At least one value, train_size or test_size, must be assigned a value or an exception will be raised.

        Args:
            train_size (float, optional): The fraction of the total dataset to use for training.
                Either train_size or test_size must be set. Defaults to None.
            test_size (float, optional): The fraction of the total dataset to use for testing.
                Either test_size of train_size must be set. Defaults to None.

        Returns:
            tuple[Dataset, Dataset]: Tuple of (train, test) datasets.

        Raises:
            ValueError: If train_size and test_size are both None.
        """
        if train_size is None and test_size is None:
            raise ValueError("Either train_size or test_size must be assigned a value. Both were None.")

        if train_size is None:
            train_size = 1.0 - test_size
        elif test_size is None:
            test_size = 1.0 - train_size

        self._ensure_tfrecords_exist()

        train_indices, test_indices = self._generate_shuffled_indices(train_size, test_size)

        raw_ds = self._load_raw_ds()
        train_ds = self._generate_filtered_ds(raw_ds, train_indices)
        test_ds = self._generate_filtered_ds(raw_ds, test_indices)

        return train_ds, test_ds

    def kfold_on_metadata(self, metadata_field: str) -> Tuple[tf.train.Feature, tf.train.Feature, str]:
        """Generate a fold based validation on a metadata field.

        For each unique value associated with the metadata field, a different train-test split will be generated.
        The test set will include all examples that have the same metadata value.
        The train set will include the remainder of the examples.
        No attempts to balance the size of the folds are made, so some training may result in very skewed sizes.
        The order of the training set is shuffled so that
        examples with different metadata values are interleaved throughout the dataset.

        Args:
            metadata_field (str): The metadata field to perform cross validation on.

        Yields:
            Iterator[Tuple[tf.train.Feature, tf.train.Feature, str]]: Tuples of (train_set, test_set, fold_value)
                where fold_value is the current value being held out for the test set.
        """
        self._ensure_tfrecords_exist()

        raw_ds = self._load_raw_ds()

        folds = {
            fold: self._get_indices_for_metadata(metadata_field, fold)
            for fold in self.metadata_stats["values"][metadata_field]
        }

        for fold in self.metadata_stats["values"][metadata_field]:
            test_indices = folds[fold]
            train_indices = list(chain(*[v for k, v in folds.items() if k != fold]))
            train_ds = self._generate_filtered_ds(raw_ds, train_indices)
            test_ds = self._generate_filtered_ds(raw_ds, test_indices)
            yield train_ds, test_ds, fold

    def analyze_index_outputs(self, outputs: List[str]) -> None:
        """
        Analyze the outputs of the index file.

        This method is called after the index file is loaded.
        It can be used to analyze the outputs of the index file and store any relevant information.

        Args:
            outputs (List[str]): List of output values taken from the index file.
        """
        return

    @abstractmethod
    def load_output_feature(self, output_index: List[str]) -> tf.train.Feature:
        """
        Using the output index, load the output feature that represents the example.

        Args:
            output_index (List[str]): List of output values taken from the index file.

        Returns:
            tf.train.Feature: The output feature that represents the example.
        """
        ...

    @abstractmethod
    def output_feature_type(self) -> Any:
        """
        Feature type of the output feature.

        Returns:
            Any: The feature type and configuration of the output feature.
        """
        ...

    def _load_index(self) -> None:
        num_examples, inputs, outputs = self._parse_index()

        self.inputs = inputs
        self.outputs = outputs
        self.num_examples = num_examples

        results = self._analyze_files([os.path.join(self._dir, "in", input_file) for input_file in inputs])
        self._validate_audio_file_set(results)
        self.input_len = int(list(results["lengths"])[0] * list(results["sampling_rates"])[0])

        self.analyze_index_outputs(outputs)

    def _parse_index(self) -> None:
        with open(os.path.join(self._dir, self._name)) as f:
            lines = list(f.readlines())

        num_examples = len(lines)

        split_lines = [[s.strip() for s in line.split(",")] for line in lines]

        inputs = [line[0] for line in split_lines]
        outputs = [line[1:] for line in split_lines]

        return num_examples, inputs, outputs

    def _analyze_files(self, files: List[Union[str, Path]]) -> Mapping[str, Any]:
        analysis = {
            "sampling_rates": set(),
            "lengths": set(),
            "bits_per_sample": set(),
            "number_of_channels": set(),
            "all_exist": True,
            "do_not_exist": [],
        }

        for file in (Path(f) for f in files):
            if file.exists():
                with open(file, "rb") as f:
                    wav_file = wave.open(f)
                    sampling_rate = wav_file.getframerate()
                    bits_per_sample = wav_file.getsampwidth() * 8
                    number_of_channels = wav_file.getnchannels()
                    length = wav_file.getnframes() / sampling_rate

                    analysis["sampling_rates"].add(sampling_rate)
                    analysis["bits_per_sample"].add(bits_per_sample)
                    analysis["number_of_channels"].add(number_of_channels)
                    analysis["lengths"].add(length)

            else:
                analysis["all_exist"] = False
                analysis["do_not_exist"].append(str(file))

        return analysis

    def _validate_audio_file_set(self, file_analysis: dict) -> None:
        """Validate that the analysis results from a set of wav files shows consistent properties.

        Args:
            file_analysis (dict): Dictionary of different properties that were analyzed
                from some collection of audio files.

        Raises:
            ValueError: If one of the files does not exist or if multiple sampling rates,
                bits per sample, number of channels, or lengths are detected.
        """
        if not file_analysis["all_exist"]:
            raise ValueError(f"The following files do not exist: {', '.join(sorted(file_analysis['do_not_exist']))}")
        if len(file_analysis["sampling_rates"]) > 1:
            sampling_rates = ", ".join([str(x) for x in sorted(file_analysis["sampling_rates"])])
            raise ValueError(f"Multiple sampling rates detected: {sampling_rates}")
        if len(file_analysis["bits_per_sample"]) > 1:
            bits_per_sample = ", ".join([str(x) for x in sorted(file_analysis["bits_per_sample"])])
            raise ValueError(f"Multiple bits per sample detected: {bits_per_sample}")
        if len(file_analysis["number_of_channels"]) > 1:
            number_of_channels = ", ".join([str(x) for x in sorted(file_analysis["number_of_channels"])])
            raise ValueError(f"Multiple number of channels detected: {number_of_channels}")
        if len(file_analysis["lengths"]) > 1:
            lengths = ", ".join([str(x) for x in sorted(file_analysis["lengths"])])
            raise ValueError(f"Multiple lengths detected (seconds): {lengths}")

    def _load_metadata(self) -> Optional[List[dict]]:
        # check if metadata exists, and if so then load it for the indices
        if self._metadata_file:
            with open(os.path.join(self._dir, self._metadata_file)) as f:
                metadata = json.load(f)
            self.metadata = metadata

            stats = {
                "fields": set(),
                "values": {}
            }

            for index in metadata:
                for field, value in metadata[index].items():
                    if not isinstance(value, str):
                        raise Exception(
                            f"Only string values are allowed in metadata. Found {value} with type {type(value)}"
                        )
                    stats["fields"].add(field)
                    if field not in stats["values"]:
                        stats["values"][field] = set()
                    stats["values"][field].add(value)

            stats["fields"] = list(stats["fields"])
            for field in stats["values"]:
                stats["values"][field] = list(stats["values"][field])

            self.metadata_stats = stats

    def _load_audio_feature(self, file_path: str) -> tf.train.Feature:
        data, _ = tf.audio.decode_wav(tf.io.read_file(file_path))
        data = tf.squeeze(data)
        return tf.train.Feature(float_list=tf.train.FloatList(value=data.numpy()))

    def _generate_shuffled_indices(self, train_size: float, test_size: float) -> Tuple[List[int], List[int]]:
        # round uses banker's rounding
        num_train_indices = min(round(self.num_examples * train_size), self.num_examples)
        num_test_indices = min(round(self.num_examples * test_size), self.num_examples)

        if num_test_indices == 0:
            num_test_indices = 1
            num_train_indices -= 1
        elif num_train_indices == 0:
            num_train_indices = 1
            num_test_indices -= 1

        if num_train_indices + num_test_indices > self.num_examples:
            raise ValueError("Train and test split overlap.")

        shuffled_indices = list(range(self.num_examples))
        self._rng.shuffle(shuffled_indices)

        train_indices = shuffled_indices[:num_train_indices]
        test_indices = shuffled_indices[-num_test_indices:]

        return train_indices, test_indices

    def _get_indices_for_metadata(self, field: str, value: Any) -> List[int]:
        indices = []
        for i in range(self.num_examples):
            input_file = self.inputs[i]
            if self.metadata[input_file][field] == value:
                indices.append(i)
        return indices

    def _ensure_tfrecords_exist(self) -> None:
        if not os.path.exists(self.tfrecord_path):
            writer = tf.io.TFRecordWriter(self.tfrecord_path)
            for index, (input_, output) in enumerate(zip(self.inputs, self.outputs)):
                feature = {
                    'index': tf.train.Feature(int64_list=tf.train.Int64List(value=[index])),
                    'a_in': self._load_audio_feature(os.path.join(self._dir, "in", input_)),
                    'a_out': self.load_output_feature(output)
                }

                example = tf.train.Example(features=tf.train.Features(feature=feature))
                writer.write(example.SerializeToString())

            writer.close()

    def _load_raw_ds(self) -> tf.data.Dataset:
        feature_description = {
            'index': tf.io.FixedLenFeature((1,), tf.int64),
            'a_in': tf.io.FixedLenFeature((self.input_len,), tf.float32),
            'a_out': self.output_feature_type()
        }

        def _parser(x: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
            result = tf.io.parse_single_example(x, feature_description)
            return result['index'], result['a_in'], result['a_out']

        shuffle_size = max(int(0.1 * self.num_examples), 1000) if self._shuffle_size is None else self._shuffle_size

        result = tf.data.TFRecordDataset(self.tfrecord_path).map(_parser)

        if shuffle_size:
            result = result.shuffle(
                shuffle_size,
                reshuffle_each_iteration=False  # this must be false, otherwise the X and y sets will not stay coupled
            )

        return result

    def _generate_filtered_ds(self, raw_ds: tf.data.Dataset, indices: List[int]) -> tf.data.Dataset:
        def _split(_: tf.Tensor, a_in: tf.Tensor, a_out: tf.Tensor, i: int) -> tf.Tensor:
            if i == 1:
                return a_in
            return a_out

        filtered_ds = self._filter_ds_on_index(raw_ds, indices)
        in_ds = filtered_ds.map(partial(_split, i=1))
        out_ds = filtered_ds.map(partial(_split, i=2))

        return tf.data.Dataset.zip((in_ds, out_ds))

    def _filter_ds_on_index(self, ds: tf.data.Dataset, indices: List[int]) -> tf.data.Dataset:
        truth = [1] * len(indices)
        lookup = tf.lookup.StaticHashTable(
            tf.lookup.KeyValueTensorInitializer(keys=indices, values=truth, key_dtype=tf.int64),
            default_value=0
        )

        def _filter(index: tf.Tensor, _: tf.Tensor, __: tf.Tensor) -> tf.Tensor:
            return (lookup.lookup(index) == 1)[0]

        return ds.filter(_filter)

    @property
    def tfrecord_path(self) -> str:
        """Path to the tfrecord file."""
        return os.path.join(self._dir, f"{self._name}.tfrecord")


class AudioDataset(BaseAudioDataset):
    """Dataset for sequence to sequence audio data."""

    def analyze_index_outputs(self, outputs: List[List[str]]) -> None:
        """
        Analyze the outputs for each index to ensure they are all the same length and sampling rate.

        Args:
            outputs (List[str]): List of output file names for each index.
        """
        results = self._analyze_files([os.path.join(self._dir, "out", output[0]) for output in outputs])
        self.output_len_ = int(list(results["lengths"])[0] * list(results["sampling_rates"])[0])

    def output_feature_type(self) -> Any:
        """Feature type of AudioDataset is a float32 FixedLenFeature of shape (output_len_,)."""
        return tf.io.FixedLenFeature((self.output_len_,), tf.float32)

    def load_output_feature(self, output_index: List[str]) -> tf.train.Feature:
        """
        Load the output feature for a given index.

        Args:
            output_index (List[str]): Output file name to load.
        """
        output_file_path = os.path.join(self._dir, "out", output_index[0])
        return self._load_audio_feature(output_file_path)


class MultilabelClassificationAudioDataset(BaseAudioDataset):
    """Dataset for multilabel classification audio data."""

    def analyze_index_outputs(self, outputs: List[List[str]]) -> None:
        """
        Analyze the outputs for each index to get the label length and ensure they are all the same.

        Args:
            outputs (List[List[str]]): List of label classifications.
        """
        self.label_length = len(outputs[0][0])

    def output_feature_type(self) -> Any:
        """Feature type of MultilabelClassificationAudioDataset is a FixedLenFeature of shape (label_length,)."""
        return tf.io.FixedLenFeature([self.label_length], tf.float32)

    def load_output_feature(self, output_index: List[str]) -> tf.train.Feature:
        """
        Load an output feature for the given index.

        Args:
            output_index (List[str]): List of label classifications.

        Returns:
            tf.train.Feature: Feature for the given index.
        """
        return tf.train.Feature(
            float_list=tf.train.FloatList(value=np.array([float(c) for c in output_index[0]], dtype="float32"))
        )


class RegressionAudioDataset(BaseAudioDataset):
    """Dataset for regression audio data."""

    def analyze_index_outputs(self, outputs: List[List[str]]) -> None:
        """
        Analyze the outputs to get the number of target regression values.

        Args:
            outputs (List[List[str]]): List of target regression values.
        """
        self.n_ = len(outputs[0])

    def output_feature_type(self) -> Any:
        """Feature type of RegressionAudioDataset is a float32 FixedLenFeature of shape (n_,)."""
        return tf.io.FixedLenFeature((self.n_,), tf.float32)

    def load_output_feature(self, output_index: List[str]) -> tf.train.Feature:
        """
        Load an output feature for the given index.

        Args:
            output_index (List[str]): List of target regression values.

        Returns:
            tf.train.Feature: Feature for the given index.
        """
        return tf.train.Feature(
            float_list=tf.train.FloatList(value=np.array([float(target) for target in output_index], dtype="float32"))
        )
