#paritally based on pytorch code by Alexander Corley
from abc import ABC, abstractmethod
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

        # create placeholders for datasets
        self.train = None
        self.validate = None
        self.test = None

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

    #generate an audio dataset & its associated tfrecords
    def generate(self, ex_per_file=2400, n_processes=multiprocessing.cpu_count()) -> None:
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

    #loads dataset from files into train, test, and val datasets
    def load(self, input_size, batch_size, train_split=0.7, val_split=0.15, test_split=0.15, shuffle_buffer=1024):
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

    #wrapper to generate TF features for dataset. TF doesn't like train.Feature without the wrapper
    #copied directly from TF example code
    def _bytes_feature(self, value):
        """Returns a bytes_list from a string / byte."""
        if isinstance(value, type(tf.constant(0))):
            value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    #return all tfrecord files matching pattern dir/name#.tfrecord
    def _get_records(self):
        return glob.glob(os.path.join(self._dir, "{0}*.tfrecord".format(self._name)))

    #load and read dataset (to be mapped)
    def _load_map(self, proto_buff, input_size, input_length):
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

    #make directories for audio generation
    def _make_dirs(self):
        #if ./_dir/ doesn't exist, make it and all others
        if not os.path.exists(self._dir):
            try:
                os.makedirs(self._dir)
                os.makedirs(os.path.join(self._dir, "in"))
                os.makedirs(os.path.join(self._dir, "out"))
            except OSError as e:
                raise
        #if ./_dir/ exists, but in and out don't (ie if remove_wav=True on previous gen)
        elif not (os.path.exists(os.path.join(self._dir, "in")) or os.path.exists(os.path.join(self._dir, "out"))):
            try:
                os.makedirs(os.path.join(self._dir, "in"))
                os.makedirs(os.path.join(self._dir, "out"))
            except OSError as e:
                raise

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

    def _load_audio_feature(self, file_path: str) -> tf.train.Feature:
        data, _ = tf.audio.decode_wav(tf.io.read_file(file_path))

        #reshape x, y from [t, 1] to [t]
        data = tf.squeeze(data)

        #tensorflow can only store float32, but WAV files are 16bit. use np methods to store as 16bit
        bytes_data = np.asarray(data).astype(np.float16).tobytes()

        return self._bytes_feature(bytes_data)

    #the opposite of generate_audio(), removes all wav files after generation for storage reasons
    def _remove_wav(self):
        try:
            shutil.rmtree(os.path.join(self._dir, "in"))
            shutil.rmtree(os.path.join(self._dir, "out"))
            os.remove(os.path.join(self._dir, self._name))
        except OSError as e:
            print("Error: {0} - {1}".format(e.filename, e.strerror))

    @abstractmethod
    def load_output_feature(self, output_index: List[str]) -> tf.train.Feature:
        pass

    def _output_feature_type(self):
        return tf.io.FixedLenFeature([], tf.string)

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

    def load_output_feature(self, output_index: List[str]) -> tf.train.Feature:
        output_file_path = os.path.join(self._dir, "out", output_index[0])
        return self._load_audio_feature(output_file_path)


class MultilabelClassificationAudioDataset(BaseAudioDataset):

    def analyze_index_outputs(self, outputs: List[str]) -> None:
        self.label_length = len(outputs[0])
        return

    def _output_feature_type(self):
        return tf.io.FixedLenFeature([self.label_length], tf.float32)

    def load_output_feature(self, output_index: List[str]) -> tf.train.Feature:
        return tf.train.Feature(float_list=tf.train.FloatList(value=np.array([float(c) for c in output_index[0]], dtype="float32")))


class RegressionAudioDataset(BaseAudioDataset):

    def analyze_index_outputs(self, outputs: List[str]) -> None:
        self.n_ = len(outputs[0].split(","))

    def _output_feature_type(self):
        return tf.io.FixedLenFeature([self.n_], tf.float32)

    def load_output_feature(self, output_index: str) -> tf.train.Feature:
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
