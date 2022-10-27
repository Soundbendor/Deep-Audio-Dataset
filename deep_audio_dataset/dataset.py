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



#dataset that allows you to go from pre-existing audio files to a sparse output (e.g. for classificatiom)
#index file should be created as such:
#input_audio,csv_features
#ex: piano1.wav,0,0,.5,1,0,0,.2,0
class SimpleAudioClassificationDataset(AudioDataset):
    def __init__(self, directory, name):
        super().__init__(directory, name)


    #no synthetic audio needs to be generated, so just make records if they don't exist
    def generate(self):
        if not self._get_records():
            self.generate_records(**kwargs)


    #loads dataset from files into train, test, and val datasets
    def load(self, input_size, batch_size, output_size, train_split=0.7, val_split=0.15, test_split=0.15, shuffle_buffer=1024):
        #load info about class from param file
        self._load_params(batch_size, train_split, val_split, test_split)

        #load serialized dataset from files, then deserialize
        raw_ds = tf.data.TFRecordDataset(self._get_records())
        proc_ds = raw_ds.map(lambda x: self._load_map(x, input_size, self._length * self._sample_rate, output_size)).shuffle(buffer_size=shuffle_buffer)

        #create train/val/test datasets from loaded main ds
        self.train = proc_ds.take(self._train_size).repeat().batch(self._batch_size).prefetch(tf.data.experimental.AUTOTUNE)
        self.test = proc_ds.skip(self._train_size)
        self.validate = self.test.skip(self._val_size).repeat().batch(self._batch_size)
        self.test = self.test.take(self._test_size).repeat().batch(self._batch_size)


    #generate tfrecord files from a set of WAV files. Target size ~=150MB each.
    def generate_records(self, ex_per_file=4800, n_processes=multiprocessing.cpu_count()):
        #generate on CPU only. If flag isn't set false, GPU likely will OOM
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

        #get list of audio file / output pairing from index file
        with open(os.path.join(self._dir, self._name)) as f:
            index = []
            reader = csv.reader(f, delimiter=',')
            for row in reader:
                data = []
                for val in row[1:]:
                    data.append(float(val))
                index.append((row[0], np.asarray(data, dtype=np.float32)))

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


    #job for multiprocessed generation of tfrecords
    def _record_generation_job(self, index, ex_per_file, offset, progress=False):
        #create first file to write to
        writer = tf.io.TFRecordWriter(os.path.join(self._dir, "{0}{1}.tfrecord".format(self._name, offset)))

        #loop through each example and load into tfrecord files
        for i, data in enumerate(index):
            #every ex_per_file create a new file. ex_per_file should be set such that the size of each tfrecord is between 100 and 200 mb
            if (i % ex_per_file) == 0 and i != 0:
                #close current writer and open new file
                writer.close()
                writer = tf.io.TFRecordWriter(os.path.join(self._dir, "{0}{1}.tfrecord".format(self._name, int(i/ex_per_file)+offset)))

            #open WAV file and convert to float arrays
            file_x = os.path.join(self._dir, "in", data[0])
            x, _ = tf.audio.decode_wav(tf.io.read_file(file_x))
            sparse = tf.sparse.from_dense(data[1])
            d = [tf.squeeze(x), sparse]          #reshape x from [t, 1] to [t]

            #tensorflow can only store float32, but WAV files are 16bit. use np methods to store as 16bit

            feature = {"a_in": self._bytes_feature(np.asarray(d[0]).astype(np.float16).tobytes()),
                      "b_ind": tf.train.Feature(int64_list=tf.train.Int64List(value=sparse.indices)),
                      "b_val": tf.train.Feature(float_list=tf.train.FloatList(value=sparse.values))}

            #create TF example for proper serialization
            example = tf.train.Example(features=tf.train.Features(feature=feature))
            writer.write(example.SerializeToString())

            #update progress
            if progress:
                print("Creating TFRecords... {:.1f}%".format(100*i/len(index)), end="\r")


    #load and read dataset (to be mapped)
    def _load_map(self, proto_buff, input_size, input_length, output_size):
        #convert from serialized to binary
        feats = {"a_in": tf.io.FixedLenFeature([], tf.string),
                 "sparse": tf.io.SparseFeature(index_key="b_ind", value_key="b_val", dtype=tf.float32, size=output_size)}
        features = tf.io.parse_single_example(proto_buff, features=feats)

        #convert from binary to floats
        features["a_in"] = tf.io.decode_raw(features["a_in"], tf.float16)

        #split into sections to feed into model
        #zero pad audio so split works
        d = [tf.pad(features["a_in"], [[0, int(input_size - (input_length % input_size))]]), features["sparse"]]
        d[0] = tf.reshape(d[0], [-1, input_size])
        d[1] = tf.sparse.to_dense(d[1])

        return d



class AudioClassificationDataset(AudioDataset):
    def __init__(self, directory, name):
        super().__init__(directory, name)


    #no synthetic audio needs to be generated, so just make records if they don't exist
    def generate(self, input_size, **kwargs):
        if not self._get_records():
            self.generate_records(input_size, **kwargs)


    #loads dataset from files into train, test, and val datasets
    def load(self, batch_size, output_size, train_split=0.7, val_split=0.15, test_split=0.15, shuffle_buffer=1024):
        #load info about class from param file
        self._load_params(batch_size, train_split, val_split, test_split)

        #load serialized dataset from files, then deserialize
        raw_ds = tf.data.TFRecordDataset(self._get_records())
        proc_ds = raw_ds.map(lambda x: self._load_map(x, output_size)).shuffle(buffer_size=shuffle_buffer)

        #create train/val/test datasets from loaded main ds
        self.train = proc_ds.take(self._train_size).repeat().padded_batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
        self.test = proc_ds.skip(self._train_size)
        self.validate = self.test.skip(self._val_size).repeat().batch(self._batch_size)
        self.test = self.test.take(self._test_size).repeat().batch(self._batch_size)


    #generate tfrecord files from a set of WAV files. Target size ~=150MB each.
    def generate_records(self, input_size, ex_per_file=16, n_processes=multiprocessing.cpu_count()):
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
                subarrays[i] = (x, input_size, ex_per_file, offset*i, True)
            else:
                subarrays[i] = (x, input_size, ex_per_file, offset*i, False)

        #spawn multiproccesses
        processes = []
        for i in subarrays:
            p = multiprocessing.Process(target=self._record_generation_job, args=i)
            p.start()
            processes.append(p)

        #wait for processes to complete
        for i in processes:
            i.join()


    #write some information about dataset to json for future loading/human reading
    def _write_params(self, input_lower, input_upper, n_steps, length, input_size, sample_rate, waves):
        params = {}

        #for class functionality
        params["size"] = n_steps * len(waves)
        params["length"] = length
        params["sample_rate"] = sample_rate
        params["input_size"] = input_size

        #for human reference
        params["input_lower"] = input_lower
        params["input_upper"] = input_upper
        params["waves"] = waves

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
        self.input_size = params["input_size"]
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



    #job for multiprocessed generation of tfrecords
    def _record_generation_job(self, index, input_size, ex_per_file, offset, progress=False):
        #create first file to write to
        writer = tf.io.TFRecordWriter(os.path.join(self._dir, "{0}{1}.tfrecord".format(self._name, offset)))

        #loop through each example and load into tfrecord files
        for i, data in enumerate(index):
            #every ex_per_file create a new file. ex_per_file should be set such that the size of each tfrecord is between 100 and 200 mb
            if (i % ex_per_file) == 0 and i != 0:
                #close current writer and open new file
                writer.close()
                writer = tf.io.TFRecordWriter(os.path.join(self._dir, "{0}{1}.tfrecord".format(self._name, int(i/ex_per_file)+offset)))

            #open WAV file and split into chunks
            file_x = os.path.join(self._dir, "in", data[0])
            x, _ = tf.audio.decode_wav(tf.io.read_file(file_x))
            x = tf.squeeze(x)                   #reshape x from [t, 1] to [t]
            x = tf.pad(x, [[0, int(input_size - (x.shape[0] % input_size))]]) #zero pad audio so split works    (zero pad now as opposed to loading)

            #read and process csv file
            csv_data = []
            with open(os.path.join(self._dir, data[1])) as f:
                reader = csv.reader(f, delimiter=',')
                next(reader, None)      #skip headers

                #iterate through csv
                for row in reader:
                    foo = []
                    for val in row[1:]:
                        foo.append(float(val))

                    csv_data.append(np.asarray(foo, dtype=np.float32))


            #tensorflow can only store float32, but WAV files are 16bit. use np methods to store as 16bit
            feature = {"a_in": self._bytes_feature(np.asarray(x).astype(np.float16).tobytes()),
                      "b_val": tf.train.Feature(float_list=tf.train.FloatList(value=tf.reshape(tf.convert_to_tensor(csv_data, dtype=tf.float32), [-1])))}

            #create TF example for proper serialization
            example = tf.train.Example(features=tf.train.Features(feature=feature))
            writer.write(example.SerializeToString())

            #update progress
            if progress:
                print("Creating TFRecords... {:.1f}%".format(100*i/len(index)), end="\r")


    #load and read dataset (to be mapped)
    def _load_map(self, proto_buff, output_size):
        #convert from serialized to binary
        feats = {"a_in": tf.io.FixedLenFeature([], tf.string),
                 "b_val": tf.io.VarLenFeature(tf.float32)}
        features = tf.io.parse_single_example(proto_buff, features=feats)

        #convert from binary to floats
        features["a_in"] = tf.io.decode_raw(features["a_in"], tf.float16)

        #reconstruct sparse tensor
        y = tf.reshape(tf.sparse.to_dense(features["b_val"]), [-1, output_size])

        #split into sections to feed into model
        x = tf.reshape(features["a_in"], [-1, self.input_size])

        return [x, y]



class PianoDataset(AudioDataset):
    def __init__(self, directory, name, size, batch_size, train_split=0.7, val_split=0.15, test_split=0.15):
        super().__init__(directory, name, size, batch_size, train_split, val_split, test_split)


    #generate an audio dataset & its associated tfrecords
    def generate(self, remove_wav=False, ex_per_file=2400, n_processes=multiprocessing.cpu_count()):
        #check if dataset has already been created. if so, do not generate again
        if not self._get_records():        #_get_records() returns a list of records. not [] == True when list is empty
            self._generate_index()
            self.generate_records(ex_per_file=ex_per_file, n_processes=n_processes)
            if remove_wav:
                self._remove_wav()
                pass


    #loads dataset from files into train, test, and val datasets
    def load(self, input_size, shuffle_buffer=1024):
        #load serialized dataset from files, then deserialize
        raw_ds = tf.data.TFRecordDataset(self._get_records())
        proc_ds = raw_ds.map(lambda x: self._load_map(x, input_size)).shuffle(buffer_size=shuffle_buffer)

        #create train/val/test datasets from loaded main ds
        #self.train = proc_ds.take(self._train_size).repeat().padded_batch(self._batch_size, padded_shapes=([None, input_size],[None, input_size])).prefetch(tf.data.experimental.AUTOTUNE)
        self.train = proc_ds.take(self._train_size).repeat().batch(self._batch_size).prefetch(tf.data.experimental.AUTOTUNE)
        self.test = proc_ds.skip(self._train_size)
        self.validate = self.test.skip(self._val_size).repeat().batch(self._batch_size)
        self.test = self.test.take(self._test_size).repeat().batch(self._batch_size)


    def _generate_index(self):
        target_file = open(os.path.join(self._dir, self._name), "w")

        self._make_dirs()

        #only get file names
        files = [_ for _ in os.walk(os.path.join(self._dir, "in"))][0][-1]
        for file in files:
            #resample to 16kHz and place resampled in out dir
            sox("sox {0} {1} rate 16000".format(os.path.join(self._dir, "in", file), os.path.join(self._dir, "out", file)).split())
            #copy each resampled file and overwrite the high sample rate file
            shutil.copy(os.path.join(self._dir, "out", file), os.path.join(self._dir, "in", file))

            #find the frequencies as numbers in the file name. most of these will be '##.' if so remove '.'
            note = file[12:15]
            if note[-1] == '.':
                note = note[:-1]

            #this means output freq isn't in dataset
            if int(note) > 96:
                continue

            #octaves are 12 semitones apart. thus have output be 12 higher
            print(file[:12] + "{0}.wav,".format(int(note)) + file[:12] + "{0}.wav".format(int(note)+12), file=target_file)

        target_file.close()


    #load and read dataset (to be mapped)
    def _load_map(self, proto_buff, input_size):
        #convert from serialized to binary
        feats = {"a_in": tf.io.FixedLenFeature([], tf.string),
                 "a_out": tf.io.FixedLenFeature([], tf.string)}
        features = tf.io.parse_single_example(proto_buff, features=feats)

        #convert from binary to floats
        features["a_in"] = tf.io.decode_raw(features["a_in"], tf.float16)
        features["a_out"] = tf.io.decode_raw(features["a_out"], tf.float16)

        #split into sections to feed into LSTM
        #zero pad values so split works
        d = [x for _, x in features.items()]
        #pad each audio file to the length of the longest audio file
        #pad by len(longest) - len(this_file). Must ensure longest is a mult of input_size though
        d = [tf.pad(x, [[0, tf.reduce_max([tf.shape(d[0]), tf.shape(d[1])]) + (input_size - (tf.reduce_max([tf.shape(d[0]), tf.shape(d[1])]) % input_size)) - tf.shape(x)[0]]]) for x in d]
        d = [tf.reshape(x, [-1, input_size]) for x in d]

        return d


    #make directories for audio generation
    def _make_dirs(self):
        #if ./_dir/ doesn't exist, make it and all others
        if not os.path.exists(os.path.join(self._dir, "out")):
            try:
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
