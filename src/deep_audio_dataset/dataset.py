#paritally based on pytorch code by Alexander Corley
from abc import ABC, abstractmethod
from datetime import datetime
from functools import partial
import glob
import json
import multiprocessing
import os
from pathlib import Path
import random
import shutil
import time
from typing import (
    Any,
    Iterable,
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

        self.input_len_ = None

    @abstractmethod
    def generate(self, *args, **kwargs) -> None:
        pass

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

    def _validate_audio_file_set(self, file_paths: List[Union[str, Path]]) -> None:
        """Validate that all of the wav files have consistent properties.

        Args:
            file_paths (list): List of file paths (as strings or Path objects) for each of the files in the set to validate.

        Raises:
            ValueError: If one of the files does not exist or if multiple sampling rates, bits per sample, number of channels, or lengths are detected.
        """
        file_analysis = self._analyze_files(file_paths)

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

    def load_metadata(self) -> Optional[List[dict]]:
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

    def generate(self, ex_per_file=2400, n_processes=multiprocessing.cpu_count()) -> None:
        """Generate an audio dataset and its associated tfrecords.

        Args:
            ex_per_file (int, optional): Target number of examples to have in each tfrecord. Defaults to 2400.
            n_processes (_type_, optional): Number of processes to use for parallelizing the tfrecord generation job. Defaults to the number of cores reported by the multiprocessing package.
        """
        #generate on CPU only. If flag isn't set false, GPU likely will OOM
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

        #get list of all audio files from index file
        with open(os.path.join(self._dir, self._name)) as f:
            index = [tuple(line.strip().split(",")) for line in f.readlines()]

        inputs = [x[0] for x in index]
        outputs = [",".join(x[1:]) for x in index]

        self.analyze_index_outputs(outputs)

        # don't check values if they don't end in "wav"
        # FIXME this is pretty spotty behaviour, need a more robust system of inspecting inputs and outputs in BaseAudioDataset
        input_file_paths = [Path(self._dir).joinpath("in/" + f) for i in index for f in i if len(f) > 0 and f.endswith("wav")]

        self._validate_audio_file_set(input_file_paths)

        self.load_metadata()

        #see if number of processes are excessive. if so, adjust to appropriate amt
        if np.ceil(len(index)/ex_per_file) < n_processes:
            n_processes = int(np.ceil(len(index)/ex_per_file))

        #shuffle the indicies to guarentee randomness across files
        self._rng.shuffle(index)

        # example_chunks = np.array_split(index, ex_per_file)
        if ex_per_file == 1:
            example_chunks = [[x] for x in index]
        else:
            example_chunks = [x for x in np.array_split(index, ex_per_file) if len(x) > 0]

        job_args = [(x, i, [self.metadata[index_name[0]] for index_name in x] if self.metadata else []) for i, x in enumerate(example_chunks) if len(x) > 0]

        with multiprocessing.Pool(n_processes) as pool:
            pool.starmap(self._record_generation_job, job_args)

    def load(self, input_size, batch_size, train_split=0.7, val_split=0.15, test_split=0.15, shuffle_buffer=1024):
        """Loads dataset from files into train, test, and val datasets.
        THIS FUNCTION IS GOING TO GET REMOVED.

        Args:
            input_size (_type_): _description_
            batch_size (_type_): _description_
            train_split (float, optional): _description_. Defaults to 0.7.
            val_split (float, optional): _description_. Defaults to 0.15.
            test_split (float, optional): _description_. Defaults to 0.15.
            shuffle_buffer (int, optional): _description_. Defaults to 1024.
        """
        #load info about class from param file
        self._load_params(batch_size, train_split, val_split, test_split)

        #load serialized dataset from files, then deserialize
        raw_ds = tf.data.TFRecordDataset(self._get_records())
        proc_ds = raw_ds.map(lambda x: self._load_map(x, input_size, self._length * self._sample_rate)).shuffle(buffer_size=shuffle_buffer)

        #create train/val/test datasets from loaded main ds
        self.train = proc_ds.take(self._train_size).repeat().batch(self._batch_size).prefetch(tf.data.experimental.AUTOTUNE)
        self.test = proc_ds.skip(self._train_size)
        self.validate = self.test.skip(self._val_size).repeat().batch(self._batch_size)
        self.test = self.test.take(self._test_size).repeat().batch(self._batch_size)

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

    def _get_records(self) -> List[str]:
        """Return all tfrecord files matching pattern dir/name.tfrecord

        Returns:
            list[str]: List of file paths.
        """
        return glob.glob(os.path.join(self._dir, "{0}*.tfrecord".format(self._name)))

    def _load_map(self, proto_buff: Any, input_size: int, input_length: int) -> List[Any]:
        """Load and read dataset (to be mapped)

        Args:
            proto_buff (any): A string tensor to parse from.
            input_size (int): _description_
            input_length (int): _description_

        Returns:
            list[any]: List of tensors.
        """
        # FIXME this function needs to be checked to make sure it makes sense
        #convert from serialized to binary
        feats = {"a_in": tf.io.FixedLenFeature([], tf.string),
                 "a_out": tf.io.FixedLenFeature([], tf.string)}
        features = tf.io.parse_single_example(proto_buff, features=feats)

        #convert from binary to floats
        features["a_in"] = tf.io.decode_raw(features["a_in"], tf.float16)
        features["a_out"] = tf.io.decode_raw(features["a_out"], tf.float32)

        #split into sections to feed into LSTM
        #zero pad values so split works
        d = [tf.pad(x, [[0, input_size - (input_length % input_size)]]) for _, x in features.items()]
        d = [tf.reshape(j, [-1, input_size]) for j in d]

        return d

    def _record_generation_job(
        self,
        index: Iterable[List[str]],
        id: int,
        metadata: Iterable[dict]
    ) -> None:
        """
        Job for multiprocessed generation of tfrecords.

        Args:
            index (iterable(list(str))): Iterable of index lists. Each element is a list of strings from the index configuration that represent an index configuration for a single example.
            id (int): ID used for tfrecord file name.
            progress (bool): Whether or not to print the progress bar. Defaults to False.
            metadata (iterable(dict)): Metadata dictionaries to also encode.
        """
        writer = tf.io.TFRecordWriter(os.path.join(self._dir, f"{self._name}{id}.tfrecord"))

        for i, files in enumerate(index):
            input_file_path = os.path.join(self._dir, "in", files[0])

            feature = {
                "a_in": self._load_audio_feature(input_file_path),
                "a_out": self.load_output_feature(files[1:])
            }

            if metadata:
                feature["metadata"] = tf.train.Feature(bytes_list=tf.train.BytesList(value=[json.dumps(metadata[i]).encode()]))

            #create TF example for proper serialization
            example = tf.train.Example(features=tf.train.Features(feature=feature))
            writer.write(example.SerializeToString())

            #update progress
            # if progress:
            #     print("Creating TFRecords... {:.1f}%".format(100*i/len(index)), end="\r")

    # def _load_audio_feature(self, file_path: str) -> tf.train.Feature:
    #     """Loads an audio file and converts it to a feature.

    #     Args:
    #         file_path (str): path to the audio file.

    #     Returns:
    #         tf.train.Feature: feature representing the audio file.
    #     """
    #     data, _ = tf.audio.decode_wav(tf.io.read_file(file_path))

    #     #reshape x, y from [t, 1] to [t]
    #     data = tf.squeeze(data)

    #     #tensorflow can only store float32, but WAV files are 16bit. use np methods to store as 16bit
    #     bytes_data = np.asarray(data).astype(np.float16).tobytes()

    #     return self._bytes_feature(bytes_data)

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

    def _generate_shuffled_indices(self, num_examples: int, train_size: float, test_size) -> Tuple[List[int], List[int]]:
        # round uses banker's rounding
        num_train_indices = min(round(num_examples * train_size), num_examples)
        num_test_indices = min(round(num_examples * test_size), num_examples)

        if num_test_indices == 0:
            num_test_indices = 1
            num_train_indices -= 1
        elif num_train_indices == 0:
            num_train_indices = 1
            num_test_indices -= 1

        if num_train_indices + num_test_indices > num_examples:
            raise ValueError("Train and test split overlap.")

        shuffled_indices = [i for i in range(num_examples)]
        self._rng.shuffle(shuffled_indices)

        train_indices = shuffled_indices[:num_train_indices]
        test_indices = shuffled_indices[-num_test_indices:]

        return train_indices, test_indices

    def _save_generated_dataset(self, name: str, inputs: List[str], outputs: List[str], indices: List[int]) -> tf.data.Dataset:
        writer = tf.io.TFRecordWriter(os.path.join(self._dir, "generated_records", f"{name}.tfrecord"))

        for i in indices:
            input_file_name = inputs[i]
            output_info = outputs[i]

            input_file_path = os.path.join(self._dir, "in", input_file_name)

            feature = {
                "a_in": self._load_audio_feature(input_file_path),
                "a_out": self.load_output_feature(output_info)
            }

            example = tf.train.Example(features=tf.train.Features(feature=feature))
            writer.write(example.SerializeToString())

        feature_description = {
            'a_in': tf.io.FixedLenFeature((self.input_len_,), tf.float32),
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

        # calculate the indices
        # find how many examples there are
        num_examples, inputs, outputs = self._load_index()

        results = self._analyze_files([os.path.join(self._dir, "in", input) for input in inputs])
        self.input_len_ = int(list(results["lengths"])[0] * list(results["sampling_rates"])[0])

        self.analyze_index_outputs(outputs)

        train_indices, test_indices = self._generate_shuffled_indices(num_examples, train_size, test_size)

        self._ensure_generated_records_directory()

        suffix = datetime.now().strftime("%Y%m%d%H%M%S")
        train_set = self._save_generated_dataset(f"train_{suffix}", inputs, outputs, train_indices)
        test_set = self._save_generated_dataset(f"test_{suffix}", inputs, outputs, test_indices)

        return train_set, test_set

    def kfold_on_metadata(self, metadata_field: str) -> Tuple[tf.train.Feature, tf.train.Feature, str]:
        feature_description = {
            'a_in': tf.io.FixedLenFeature([], tf.string),
            'a_out': self._output_feature_type(),
            'metadata': tf.io.VarLenFeature(tf.string)
        }

        def _parser(x):
            return tf.io.parse_single_example(x, feature_description)

        raw_ds = tf.data.TFRecordDataset(list(Path(self._dir).glob("*.tfrecord")))
        parsed_ds = raw_ds.map(_parser)

        for value in self.metadata_stats["values"][metadata_field]:
            # grab the tfrecords and map them with a filter
            training_set = []
            test_set = []
            for x in parsed_ds:
                metadata = json.loads(x["metadata"].values.numpy()[0].decode())
                if metadata[metadata_field] == value:
                    test_set.append([x["a_in"], x["a_out"]])
                else:
                    training_set.append([x["a_in"], x["a_out"]])

            yield tf.data.Dataset.from_tensor_slices(training_set), tf.data.Dataset.from_tensor_slices(test_set), value


class AudioDataset(BaseAudioDataset):

    def analyze_index_outputs(self, outputs: List[List[str]]) -> None:
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

    def analyze_index_outputs(self, outputs: List[str]) -> None:
        self.n_ = len(outputs[0].split(","))

    def _output_feature_type(self):
        return tf.io.FixedLenFeature((self.n_,), tf.float32)

    def load_output_feature(self, output_index: List[str]) -> tf.train.Feature:
        return tf.train.Feature(float_list=tf.train.FloatList(value=np.array([float(target) for target in output_index], dtype="float32")))

    def kfold_on_metadata(self, metadata_field: str):
        feature_description = {
            'a_in': tf.io.FixedLenFeature([], tf.string),
            'a_out': self._output_feature_type(),
            'metadata': tf.io.VarLenFeature(tf.string)
        }

        def _parser(x):
            return tf.io.parse_single_example(x, feature_description)

        raw_ds = tf.data.TFRecordDataset(list(Path(self._dir).glob("*.tfrecord")))
        parsed_ds = raw_ds.map(_parser)

        for value in self.metadata_stats["values"][metadata_field]:
            # grab the tfrecords and map them with a filter
            x_training_set = []
            y_training_set = []
            x_test_set = []
            y_test_set = []

            for x in parsed_ds:
                metadata = json.loads(x["metadata"].values.numpy()[0].decode())
                if metadata[metadata_field] == value:
                    x_test_set.append([np.frombuffer(x["a_in"].numpy(), dtype="float16")])
                    y_test_set.append(tf.reshape(x["a_out"], (1, self.n_)))
                else:
                    x_training_set.append([np.frombuffer(x["a_in"].numpy(), dtype="float16")])
                    y_training_set.append(tf.reshape(x["a_out"], (1, self.n_)))

            training_set = tf.data.Dataset.from_tensor_slices((x_training_set, y_training_set))
            test_set = tf.data.Dataset.from_tensor_slices((x_test_set, y_test_set))

            yield training_set, test_set, value
