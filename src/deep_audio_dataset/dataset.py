#paritally based on pytorch code by Alexander Corley
from abc import ABC, abstractmethod
from datetime import datetime
from functools import partial
import json
import multiprocessing
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
    def __init__(self, directory: str, index_file: str, seed: Optional[Any] = None, metadata_file: Optional[str] = None):
        """Initialize the BaseAudioDataset.

        Args:
            directory (str): Base directory for the dataset to use.
            index_file (str): Name of the data index file.
            seed (any, optional): Seed to use for the random number generator.
        """
        if seed is None:
            self._rng = random.Random(time.time())
        else:
            self._rng = random.Random(seed)

        #store args as class members
        self._dir = directory
        self._name = index_file

        self._metadata_file = metadata_file
        self.metadata = None
        self.metadata_stats = None

        self.inputs = None
        self.outputs = None
        self.input_len = None
        self.num_examples = None

        self._load_index()
        self._load_metadata()

    def train_test_split(
        self,
        train_size: Optional[float] = None,
        test_size: Optional[float] = None
    ) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
        """Split the dataset into two disjoint subsets: train set and test set.

        While the split does not have to cover the entire dataset, the sizes cannot be larger than the whole dataset (i.e. this must be true: train_size + test_size <= 0).
        At least one value, train_size or test_size, must be assigned a value or an exception will be raised.

        Args:
            train_size (float, optional): The fraction of the total dataset to use for training. Either train_size or test_size must be set. Defaults to None.
            test_size (float, optional): The fraction of the total dataset to use for testing. Either test_size of train_size must be set. Defaults to None.

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

        train_indices, test_indices = self._generate_shuffled_indices(train_size, test_size)

        self._ensure_generated_records_directory()

        suffix = datetime.now().strftime("%Y%m%d%H%M%S")
        train_set = self._save_generated_dataset(f"train_{suffix}", train_indices)
        test_set = self._save_generated_dataset(f"test_{suffix}", test_indices)

        return train_set, test_set

    def kfold_on_metadata(self, metadata_field: str) -> Tuple[tf.train.Feature, tf.train.Feature, str]:
        """Generate a fold based validation on a metadata field.

        For each unique value associated with the metadata field, a different train-test split will be generated.
        The test set will include all examples that have the same metadata value.
        The train set will include the remainder of the examples.
        No attempts to balance the size of the folds are made, so some training may result in very skewed sizes.
        The order of the training set is shuffled so that examples with different metadata values are interleaved throughout the dataset.

        Args:
            metadata_field (str): The metadata field to perform cross validation on.

        Yields:
            Iterator[Tuple[tf.train.Feature, tf.train.Feature, str]]: Tuples of (train_set, test_set, fold_value) where fold_value is the current value being held out for the test set.
        """
        # create a tfrecord for each fold
        suffix = datetime.now().strftime("%Y%m%d%H%M%S")
        folds = {
            fold: self._save_generated_dataset(
                f"{suffix}_{fold}.tfrecord",
                self._get_indices_for_metadata(metadata_field, fold)
            ) for fold in self.metadata_stats["values"][metadata_field]
        }

        for fold in self.metadata_stats["values"][metadata_field]:
            test_set = folds[fold]
            train_set = tf.data.Dataset.sample_from_datasets([
                v for k, v in folds.items() if k != fold
            ])
            yield train_set, test_set, fold

    def _load_index(self) -> None:
        num_examples, inputs, outputs = self._parse_index()

        self.inputs = inputs
        self.outputs = outputs
        self.num_examples = num_examples

        results = self._analyze_files([os.path.join(self._dir, "in", input) for input in inputs])
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

        for file in files:
            file = Path(file)
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
            file_analysis (dict): Dictionary of different properties that were analyzed from some collection of audio files.

        Raises:
            ValueError: If one of the files does not exist or if multiple sampling rates, bits per sample, number of channels, or lengths are detected.
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

        return

    def analyze_index_outputs(self, outputs: List[str]) -> None:
        return

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
                        raise Exception(f"Only string values are allowed in metadata. Found {value} with type {type(value)}")
                    stats["fields"].add(field)
                    if field not in stats["values"]:
                        stats["values"][field] = set()
                    stats["values"][field].add(value)

            stats["fields"] = list(stats["fields"])
            for field in stats["values"]:
                stats["values"][field] = list(stats["values"][field])

            self.metadata_stats = stats

    def _bytes_feature(self, value: bytes) -> tf.train.Feature:
        """Returns a bytes_list from a string / byte.
        Wrapper to generate TF features for dataset.
        TF doesn't like train.Feature without the wrapper
        Copied directly from TF example code

        Args:
            value (bytes): Bytes string to convert to a bytes list feature.

        Returns:
            Feature: bytes list feature.
        """
        if isinstance(value, type(tf.constant(0))):
            value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    def _load_audio_feature(self, file_path: str) -> tf.train.Feature:
        data, _ = tf.audio.decode_wav(tf.io.read_file(file_path))
        data = tf.squeeze(data)
        return tf.train.Feature(float_list=tf.train.FloatList(value=data.numpy()))

    @abstractmethod
    def load_output_feature(self, output_index: List[str]) -> tf.train.Feature:
        pass

    def _output_feature_type(self):
        return tf.io.FixedLenFeature([], tf.string)

    def _ensure_generated_records_directory(self) -> None:
        if not os.path.exists(os.path.join(self._dir, "generated_records")):
            os.mkdir(os.path.join(self._dir, "generated_records"))

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

        shuffled_indices = [i for i in range(self.num_examples)]
        self._rng.shuffle(shuffled_indices)

        train_indices = shuffled_indices[:num_train_indices]
        test_indices = shuffled_indices[-num_test_indices:]

        return train_indices, test_indices

    def _save_generated_dataset(self, name: str, indices: List[int]) -> tf.data.Dataset:
        self._ensure_generated_records_directory()
        writer = tf.io.TFRecordWriter(os.path.join(self._dir, "generated_records", f"{name}.tfrecord"))

        for i in indices:
            input_file_name = self.inputs[i]
            output_info = self.outputs[i]

            input_file_path = os.path.join(self._dir, "in", input_file_name)

            feature = {
                "a_in": self._load_audio_feature(input_file_path),
                "a_out": self.load_output_feature(output_info)
            }

            example = tf.train.Example(features=tf.train.Features(feature=feature))
            writer.write(example.SerializeToString())

        feature_description = {
            'a_in': tf.io.FixedLenFeature((self.input_len,), tf.float32),
            'a_out': self._output_feature_type()
        }

        def _parser(x, key):
            result = tf.io.parse_single_example(x, feature_description)
            return result[key]

        raw_ds = tf.data.TFRecordDataset(os.path.join(self._dir, "generated_records", f"{name}.tfrecord"))
        in_ds = raw_ds.map(partial(_parser, key="a_in"))
        out_ds = raw_ds.map(partial(_parser, key="a_out"))

        return tf.data.Dataset.zip((in_ds, out_ds))

    def _load_index(self) -> Tuple[int, List[str], List[str]]:
        with open(os.path.join(self._dir, self._name)) as f:
            lines = list(f.readlines())

        num_examples = len(lines)

        split_lines = [[s.strip() for s in line.split(",")] for line in lines]

        inputs = [line[0] for line in split_lines]
        outputs = [line[1:] for line in split_lines]

        return num_examples, inputs, outputs


    def _get_indices_for_metadata(self, field: str, value: Any) -> List[int]:
        indices = []
        for i in range(self.num_examples):
            input = self.inputs[i]
            if self.metadata[input][field] == value:
                indices.append(i)
        self._rng.shuffle(indices)
        return indices


class AudioDataset(BaseAudioDataset):

    def analyze_index_outputs(self, outputs: List[List[str]]) -> None:
        print(outputs)
        results = self._analyze_files([os.path.join(self._dir, "out", output[0]) for output in outputs])
        self.output_len_ = int(list(results["lengths"])[0] * list(results["sampling_rates"])[0])

    def _output_feature_type(self):
        return tf.io.FixedLenFeature((self.output_len_,), tf.float32)

    def load_output_feature(self, output_index: List[str]) -> tf.train.Feature:
        output_file_path = os.path.join(self._dir, "out", output_index[0])
        return self._load_audio_feature(output_file_path)


class MultilabelClassificationAudioDataset(BaseAudioDataset):

    def analyze_index_outputs(self, outputs: List[List[str]]) -> None:
        self.label_length = len(outputs[0][0])
        return

    def _output_feature_type(self):
        return tf.io.FixedLenFeature([self.label_length], tf.float32)

    def load_output_feature(self, output_index: List[str]) -> tf.train.Feature:
        return tf.train.Feature(float_list=tf.train.FloatList(value=np.array([float(c) for c in output_index[0]], dtype="float32")))


class RegressionAudioDataset(BaseAudioDataset):

    def analyze_index_outputs(self, outputs: List[List[str]]) -> None:
        self.n_ = len(outputs[0])

    def _output_feature_type(self):
        return tf.io.FixedLenFeature((self.n_,), tf.float32)

    def load_output_feature(self, output_index: List[str]) -> tf.train.Feature:
        return tf.train.Feature(float_list=tf.train.FloatList(value=np.array([float(target) for target in output_index], dtype="float32")))
