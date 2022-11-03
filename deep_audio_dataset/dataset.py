#paritally based on pytorch code by Alexander Corley
from abc import ABC, abstractmethod
import csv
import glob
import json
import multiprocessing
import os
import random
import shutil
import time
from typing import Any, Optional

import numpy as np
from sox.core import sox
import tensorflow as tf


'''
#Intended API for AudioDataset
#MUST WRAP CODE IN A if __name__ == "__main__" BLOCK DUE TO MULTIPROCESSING OR IT WON'T WORK!
dataset = AudioDataset(directory, name)

#Generate dataset if it doesn't already exist
dataset.generate(lower_input_frequency, upper_input_frequency, num_files_per_wave)

#dataset.generate() will call generate_audio() and generate_records()
#however these will be left as separate functions in case a different
#audio dataset is to be used, so tfrecord files can still be used

#Load the dataset so we can use them
dataset.load(input_size, batch_size)

#Access dataset as members of the class
dataset.train
dataset.validate
dataset.test
'''


class BaseAudioDataset(ABC):
    def __init__(self, seed: Optional[Any] = None):
        # create placeholders for datasets
        self.train = None
        self.validate = None
        self.test = None

        if seed is None:
            self._rng = random.Random(time.time())
        else:
            self._rng = random.Random(seed)
    
    @abstractmethod
    def generate(self, *args, **kwargs) -> None:
        pass


class AudioDataset(BaseAudioDataset):
    def __init__(self, directory, name, seed: Optional[Any] = None):
        super().__init__(seed)

        #store args as class members
        self._dir = directory
        self._name = name


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


    #generate an audio dataset & its associated tfrecords
    def generate(self, ex_per_file=2400, n_processes=multiprocessing.cpu_count()) -> None:
        #generate on CPU only. If flag isn't set false, GPU likely will OOM
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

        #get list of all audio files from index file
        with open(os.path.join(self._dir, self._name)) as f:
            index = [tuple(i.strip().split(",")) for i in f.readlines()]

        #see if number of processes are excessive. if so, adjust to appropriate amt
        if np.ceil(len(index)/ex_per_file) < n_processes:
            n_processes = int(np.ceil(len(index)/ex_per_file))

        #offset for consistent output file numbering across processes
        offset = int((len(index)/ex_per_file)/n_processes)+1

        #if files are smaller than ex_per_file, have offset be at least 1 to prevent files from being overwritten
        if offset == 0:
            offset = 1

        #shuffle the indicies to guarentee randomness across files
        self._rng.shuffle(index)

        #create arguments to pass to subtasks
        subarrays = np.array_split(index, n_processes)
        for i, x in enumerate(subarrays):
            if i == 0:
                subarrays[i] = (x, ex_per_file, offset*i, True)
            else:
                subarrays[i] = (x, ex_per_file, offset*i, False)

        #spawn multiproccesses
        processes = []
        for i in subarrays:
            p = multiprocessing.Process(target=self._record_generation_job, args=i)
            p.start()
            processes.append(p)

        #wait for processes to complete
        for i in processes:
            i.join()
            if i.exitcode != 0:
                raise Exception(f"Subprocess in generate_records returned a non-zero exit code ({i.exitcode}), check output for additional information.")


    #write some information about dataset to json for future loading/human reading
    def _write_params(self, input_lower, input_upper, n_steps, length, sample_rate, waves, mods):
        params = {}

        #for class functionality
        params["size"] = n_steps * len(waves) * len(mods)
        params["length"] = length
        params["sample_rate"] = sample_rate

        #for human reference
        params["input_lower"] = input_lower
        params["input_upper"] = input_upper
        params["waves"] = waves
        params["mods"] = mods

        #write to json
        with open(os.path.join(self._dir, "{0}.txt".format(self._name)), "w") as file:
            json.dump(params, file)


    #load some dataset parameters from saved json file
    def _load_params(self, batch_size, train_split, val_split, test_split):
        #load params from file and decode json
        file = open(os.path.join(self._dir, "{0}.txt".format(self._name)), "r")
        try:
            params = json.load(file)
        except:
            print("Error loading json!")
            file.close()
            raise

        file.close()

        #assign values from json
        self._batch_size = batch_size
        self._length = params["length"]
        self._sample_rate = params["sample_rate"]
        self._size = params["size"]

        #ensure that the splits sum to 1 and find n_entries per sub-dataset
        if not ((train_split + val_split + test_split) == 1): raise ValueError("Data splits must sum to 1!")
        self._train_size = int(train_split * self._size)
        self._val_size = int(val_split * self._size)
        self._test_size = int(test_split * self._size)

        #steps per epoch for each sub-dataset
        self.train_steps = int(self._train_size / self._batch_size)
        self.val_steps = int(self._val_size / self._batch_size)
        self.test_steps = int(self._test_size / self._batch_size)


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
        features["a_out"] = tf.io.decode_raw(features["a_out"], tf.float16)

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


    #job for multiprocessed generation of tfrecords
    def _record_generation_job(self, index, ex_per_file, offset, progress=False):
        #create first file to write to
        writer = tf.io.TFRecordWriter(os.path.join(self._dir, "{0}{1}.tfrecord".format(self._name, offset)))

        #loop through each example and load into tfrecord files
        for i, files in enumerate(index):
            #every ex_per_file create a new file. ex_per_file should be set such that the size of each tfrecord is between 100 and 200 mb
            if (i % ex_per_file) == 0 and i != 0:
                #close current writer and open new file
                writer.close()
                writer = tf.io.TFRecordWriter(os.path.join(self._dir, "{0}{1}.tfrecord".format(self._name, int(i/ex_per_file)+offset)))

            #open WAV files and convert to float arrays
            file_x, file_y = os.path.join(self._dir, "in", files[0]), os.path.join(self._dir, "out", files[1]) #get file paths
            x, _, y, _ = *tf.audio.decode_wav(tf.io.read_file(file_x)), *tf.audio.decode_wav(tf.io.read_file(file_y))
            d = [x, y]
            d = [tf.squeeze(j) for j in d]    #reshape x, y from [t, 1] to [t]

            #tensorflow can only store float32, but WAV files are 16bit. use np methods to store as 16bit
            feature = {"a_in": self._bytes_feature(np.asarray(d[0]).astype(np.float16).tobytes()),
                       "a_out": self._bytes_feature(np.asarray(d[1]).astype(np.float16).tobytes())}

            #create TF example for proper serialization
            example = tf.train.Example(features=tf.train.Features(feature=feature))
            writer.write(example.SerializeToString())

            #update progress
            if progress:
                print("Creating TFRecords... {:.1f}%".format(100*i/len(index)), end="\r")


    #the opposite of generate_audio(), removes all wav files after generation for storage reasons
    def _remove_wav(self):
        try:
            shutil.rmtree(os.path.join(self._dir, "in"))
            shutil.rmtree(os.path.join(self._dir, "out"))
            os.remove(os.path.join(self._dir, self._name))
        except OSError as e:
            print("Error: {0} - {1}".format(e.filename, e.strerror))
