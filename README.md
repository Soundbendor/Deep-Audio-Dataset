# Audio Dataset
### A dataset for loading in audio file pairs for Tensorflow 2.0

#### Requirements
* Python 3
* Tensorflow 2.0+
* Numpy
* Sox (the program, and python package 'pip install sox')

#### Example usage

Import packages and set up hyperparams

```python
from dataset import AudioDataset

#hyperparams
input_size=256                                  #used for data loading
input_length = 16000*1                          #sample rate * length (sec), used for data loading
batch_size=64                                   #used for data loading

#dataset params
input_lower = 100                               #lowest frequency for dataset generation
input_upper = 1000                              #highest frequency for dataset generation
n_steps = 1000                                  #used for dataset generation
waves = ["sin", "square", "saw", "triangle"]    #types of waveforms used for dataset generation
dataset_size = len(waves)*n_steps               #total size of dataset, used for loading dataset
audio_length = 1                                #length of audio in seconds (used for dataset generation)
```

Create dataset object and generate data

```python
if __name__ == "__main__":
    dataset = AudioDataset("./data", "dataset", dataset_size, batch_size, train_split=0.7, val_split=0.15, test_split=0.15)

    #Generate dataset if it doesn't already exist
    dataset.generate(input_lower, input_upper, n_steps, length, remove_wav=False)

    #Load the dataset so we can use them
    dataset.load(input_size, input_length)
```

Various parameters can be accessed as class members

```python
#Access dataset as members of the class
dataset.train
dataset.validate
dataset.test

#Acces the number of steps per epoch for each dataset
dataset.train_steps
dataset.val_steps
dataset.test_steps
```

Example of training and testing with this API

```python
model = tf.keras.Sequential()
#Define model here

model.fit(dataset.train, validation_data=dataset.validate, validation_steps=dataset.val_steps, epochs=epochs, steps_per_epoch=dataset.train_steps)
model.evaluate(dataset.test, steps=dataset.test_steps)
```

#### Automatic dataset generation, and using custom datasets
When `dataset.generate()` is called, it does two things. First, it will generate `dataset_size` wav files using sox, as well a csv file used later in dataset generation. These files will range in frequency from `input_lower` to `input_upper` for the input to the network (X), and `dialation*input_lower` to `dialation*input_upper` for the target of the network (Y). For each wave type in `waves`, there will be `n_steps` wav files generated. These are generated with an exponential separation in frequency in order to have the dataset distribution better mirror human psychoacoustics. All audio files are generated at 16bit precision at 16kHz.

After the wav files are generated, the program generates tfrecord files to be used with the Tensorflow 2 Dataset API. Essentially, it takes the hundreds/thousands of wav files and packages them together into bigger (~150MB) files that work well with Tensorflow. These files are then what is used to load the dataset. After the tfrecord files are generated, the original wav files can be removed using the argument `remove_wav=True`.

If you want to use this API with your own audio files, you must first generate tfrecords. Create a directory for your data and a name for your dataset, and set these as the first two arguments in `AudioDataset("./data", "dataset" ...)`. In `./data` (or your directory name), create two folders, `./data/in` and `./data/out`. In the `in` directory, place all of the data you want to input into the network (the X), and in `out` place all the data you want the network to predict (the Y). Then, to tell the program what input goes with what output (X, Y) pair, create a csv file `./data/dataset` with no extension. In this file, list the file names of the (X, Y) pairs, separated by a comma, with new pairs on new lines.
```
sin5000_00.wav,sin10000_00.wav
sin4980_48.wav,sin9960_96.wav
sin4961_03.wav,sin9922_06.wav
```
Then, call `dataset.generate_records(ex_per_file=ex_per_file, n_processes=n_processes)`. Default arguments should work. `n_processes` is how many processes should it run simultaneously to create the tfrecord files faster. Default is the number of availible CPU cores. `ex_per_file` allows you to control how many (X, Y) pairs there should be per tfrecord file. This number should be set so the files are between 150 and 200 MB. Default of 2400 should equate to this if your audio is 16 bit, 16kHz, and 1 sec in length. ( (n_bit/8) * n_samples * 2 * ex_per_file).

#### Things to note
* Code using dataset generation must be wrapped in a `if __name__ == "__main__"` block due to the use of multithreading.
* To prevent GPU OOM errors, the GPU is disabled when generating tfrecord files. Make sure you enable GPU usage after calls that would generate tfrecord files (or run the program a second time after records are generated).
* If `dataset.test_steps` or `dataset.val_steps` is less than `batch_size`, there may be a runtime error upon validation/evaluation. Increasing the amount of data in the network or increasing the train/val split size is a remedy.
