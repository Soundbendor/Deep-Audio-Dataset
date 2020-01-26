#paritally based on pytorch code by Alexander Corley
import tensorflow as tf
import numpy as np
from sox.core import sox

import os, glob, shutil
import multiprocessing


'''
#Intended API for AudioDataset
#MUST WRAP CODE IN A if __name__ == "__main__" BLOCK DUE TO MULTIPROCESSING OR IT WON'T WORK!
dataset = AudioDataset(directory, name, size, batch_size, train_split=0.7, val_split=0.15, test_split=0.15)

#Generate dataset if it doesn't already exist
dataset.generate(remove_wav=True, ex_per_file=2400, n_proccesses=4)

#dataset.generate() will call generate_audio() and generate_records()
#however these will be left as separate functions in case a different
#audio dataset is to be used, so tfrecord files can still be used

#Load the dataset so we can use them
dataset.load(input_size, input_length)

#Access dataset as members of the class
dataset.train
dataset.validate
dataset.test
'''


class AudioDataset:
    def __init__(self, directory, name, size, batch_size, train_split=0.7, val_split=0.15, test_split=0.15):
        #store args as class members
        self._dir = directory
        self._name = name
        self._batch_size = batch_size

        #ensure that the splits sum to 1 and find n_entries per sub-dataset
        if not ((train_split + val_split + test_split) == 1): raise ValueError("Data splits must sum to 1!")
        self._train_size = int(train_split * size)
        self._val_size = int(val_split * size)
        self._test_size = int(test_split * size)

        #steps per epoch for each sub-dataset
        self.train_steps = int(self._train_size / batch_size)
        self.val_steps = int(self._val_size / batch_size)
        self.test_steps = int(self._test_size / batch_size)

        #create placeholders for datasets
        self.train = None
        self.validate = None
        self.test = None


    #loads dataset from files into train, test, and val datasets
    def load(self, input_size, input_length, shuffle_buffer=1024):
        #load serialized dataset from files, then deserialize
        raw_ds = tf.data.TFRecordDataset(self._get_records())
        proc_ds = raw_ds.map(lambda x: self._load_map(x, input_size, input_length)).shuffle(buffer_size=shuffle_buffer)

        #create train/val/test datasets from loaded main ds
        self.train = proc_ds.take(self._train_size).repeat().batch(self._batch_size).prefetch(tf.data.experimental.AUTOTUNE)
        self.test = proc_ds.skip(self._train_size)
        self.validate = self.test.skip(self._val_size).repeat().batch(self._batch_size)
        self.test = self.test.take(self._test_size).repeat().batch(self._batch_size)


    #generate an audio dataset & its associated tfrecords
    def generate(self, input_lower, input_upper, n_steps, length, dialation=2, waves=["sin", "square", "saw", "triangle"], vol=0.1, remove_wav=True, ex_per_file=2400, n_processes=multiprocessing.cpu_count()):
        #check if dataset has already been created. if so, do not generate again
        if not self._get_records():        #_get_records() returns a list of records. not [] == True when list is empty
            self.generate_audio(input_lower, input_upper, n_steps, length, dialation=dialation, waves=["sin", "square", "saw", "triangle"], vol=0.1, n_processes=n_processes)
            self.generate_records(ex_per_file=ex_per_file, n_processes=n_processes)
            if remove_wav:
                self._remove_wav()
                pass


    #generate audio for a dataset, to be converted to tfrecords
    def generate_audio(self, input_lower, input_upper, n_steps, length, dialation=2, waves=["sin", "square", "saw", "triangle"], vol=0.1, n_processes=multiprocessing.cpu_count()):
        #make directories to store wav files
        self._make_dirs()

        #create list of notes to generate
        notes = []
        index_file = open(os.path.join(self._dir, self._name), "w")
        for wave in waves:
            for i in self._generate_note_list(input_lower, input_upper, n_steps):
                #generate target file paths
                #input: wave=sin, f=100.431 output: sin100_43.wav
                in_file, out_file = [wave + "{:.2f}".format(freq).replace(".", "_") + ".wav" for freq in [i, i*dialation]]
                in_sin, out_sin = [wave + " {}".format(freq) for freq in [i, i*dialation]]

                #add in and out notes to queue to be generated
                notes.append("sox -n -b 16 {0}/{1} rate 16000 synth {2} {3} vol {4} remix - ".format(
                          os.path.join(self._dir, "in"), in_file, length, in_sin, vol).split())
                notes.append("sox -n -b 16 {0}/{1} rate 16000 synth {2} {3} vol {4} remix -".format(
                          os.path.join(self._dir, "out"), out_file, length, out_sin, vol).split())

                #write to data index file (for tfrecord generation)
                print("{},{}".format(in_file, out_file), file=index_file)

        #generate data
        index_file.close()
        p = multiprocessing.Pool(n_processes)
        p.map(sox, notes)


    #generate tfrecord files from a set of WAV files. Target size ~=150MB each.
    def generate_records(self, ex_per_file=2400, n_processes=multiprocessing.cpu_count()):
        #generate on CPU only. If flag isn't set false, GPU likely will OOM
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

        #get list of all audio files from index file
        with open(os.path.join(self._dir, self._name)) as f:
            index = [tuple(i.strip().split(",")) for i in f.readlines()]

        print(len(index))
        #see if number of processes are excessive. if so, adjust to appropriate amt
        if np.ceil(len(index)/ex_per_file) < n_processes:
            n_processes = int(np.ceil(len(index)/ex_per_file))

        print(n_processes)
        #offset for consistent output file numbering across processes
        offset = int((len(index)/ex_per_file)/n_processes)+1

        #if files are smaller than ex_per_file, have offset be at least 1 to prevent files from being overwritten
        if offset == 0:
            offset = 1

        print(offset)
        #create arguments to pass to subtasks
        subarrays = np.array_split(index, n_processes)
        for i, x in enumerate(subarrays):
            print(x.shape)
            if i == 0:
                subarrays[i] = (x, ex_per_file, offset*i, True)
            else:
                subarrays[i] = (x, ex_per_file, offset*i, False)
            print(offset*i)

        #spawn multiproccesses
        processes = []
        for i in subarrays:
            p = multiprocessing.Process(target=self._record_generation_job, args=i)
            p.start()
            processes.append(p)

        #wait for processes to complete
        for i in processes:
            i.join()


    #wrapper to generate TF features for dataset. TF doesn't like train.Feature without the wrapper
    #copied directly from TF example code
    def _bytes_feature(self, value):
        """Returns a bytes_list from a string / byte."""
        if isinstance(value, type(tf.constant(0))):
            value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


    #generate notes that are exponentially separated in distance
    #this ensures that each range that we "hear" has the same ammt of data
    def _generate_note_list(self, upper, lower, steps):
        steps=steps-1
        base=(upper/lower)**(1/steps)
        return [lower*(base**i) for i in range(steps+1)]


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
        d = [tf.pad(x, [[0, input_length % input_size]]) for _, x in features.items()]
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
