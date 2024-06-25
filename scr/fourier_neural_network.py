import os
import sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from copy import copy
import multiprocessing
from multiprocessing.pool import Pool
import random
import time
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Input
from keras.callbacks import Callback
import matplotlib.pyplot as plt
from scipy.io.wavfile import write
from multiprocessing import Process, Queue

from scr import utils
from scr.utils import midi_to_freq
# print(tf.__version__)

class MyCallback(Callback):
    def __init__(self, queue:Queue=None, test_data=None, quiet=False):
        # print(tf.__version__)
        super().__init__()
        self.queue:Queue = queue
        self.test_data = test_data
        self.quiet = quiet

    def on_epoch_begin(self, epoch, logs=None):
        self.timestamp=time.perf_counter_ns()
        if not self.quiet:
            print(f"EPOCHE {epoch+1} STARTED")

    def on_epoch_end(self, epoch, logs=None):
        if self.queue:
            self.queue.put(self.model.predict(self.test_data))
        if not self.quiet:
            time_taken=time.perf_counter_ns()-self.timestamp
            print(f"EPOCHE {epoch+1} ENDED, Logs: {logs}. Time Taken: {time_taken/1_000_000_000}s")



class FourierNN:
    RANGE = 44100

    ITTERRATIONS = 5
    EPOCHS_PER_ITTERRATIONS = 1

    EPOCHS = 10

    SIGNED_RANDOMNES = 0.000000001
    DEFAULT_FORIER_DEGREE = 10
    FORIER_DEGREE_DIVIDER = 50
    FORIER_DEGREE_OFFSET = 1

    def __getstate__(self):
        # Save the model to a file before serialization
        self.current_model.save('./tmp/tmp_model.keras')
        model = self.current_model
        del self.current_model
        fourier_nn_dict = self.__dict__.copy()
        self.current_model = model
        return fourier_nn_dict

    def __setstate__(self, state):
        # Load the model from a file after deserialization
        self.__dict__.update(state)
        self.current_model = tf.keras.models.load_model('./tmp/tmp_model.keras')

    def load_tmp_model(self):
        self.current_model = tf.keras.models.load_model('./tmp/tmp_model.keras')

    def __init__(self, data=None):
        self.models:list = []
        self.current_model:keras.Model = None
        self.fourier_degree=self.DEFAULT_FORIER_DEGREE
        self.prepared_data=None
        if data is not None:
            self.prepared_data=self.prepare_data(list(data))
            self.create_new_model()

    def create_new_model(self):
        if self.current_model:
            self.models.append(self.current_model)
        self.current_model=self.create_model((self.prepared_data[1].shape[1],))

    def get_models(self)->list[Sequential]:
        return [self.current_model]+self.models

    def update_data(self, data):
        self.prepared_data=self.prepare_data(list(data))
        if not self.current_model:
            self.create_new_model()

    def get_trainings_data(self):
        return list(zip(self.prepared_data[0], self.prepared_data[2]))

    @staticmethod
    def fourier_basis(x, n=DEFAULT_FORIER_DEGREE):
        basis = [np.sin(i * x) for i in range(1, n+1)]
        basis += [np.cos(i * x) for i in range(1, n+1)]
        return np.array(basis)

    @staticmethod
    def taylor_basis(x, n=DEFAULT_FORIER_DEGREE):
        basis = [x**i / np.math.factorial(i) for i in range(n)]
        return np.array(basis)

    def prepare_data(self, data):
        self.fourier_degree = (
            len(data) // self.FORIER_DEGREE_DIVIDER) + self.FORIER_DEGREE_OFFSET
        
        actualData_x, actualData_y = zip(*data)

        # random data filling
        # if len(data) < self.RANGE:
        #     if len(data) == self.RANGE / 2:
        #         data.extend(data)
        #     else:
        #         multi = self.RANGE // len(data)
        #         rest = self.RANGE % len(data)
        #         data.extend([(dat[0] + random.uniform(-self.SIGNED_RANDOMNES, self.SIGNED_RANDOMNES),
        #                            dat[1] + random.uniform(-self.SIGNED_RANDOMNES, self.SIGNED_RANDOMNES)) for dat in data * multi])
        #         for i in range(rest):
        #             rand_x = random.uniform(-self.SIGNED_RANDOMNES,
        #                                     self.SIGNED_RANDOMNES)
        #             rand_y = random.uniform(-self.SIGNED_RANDOMNES,
        #                                     self.SIGNED_RANDOMNES)
        #             data.append(
        #                 (data[i][0] + rand_x, data[i][1] + rand_y))

        # "math" data filling
        data=sorted(data, key=lambda x: x[0])
        while len(data) < self.RANGE:
            for a, b in utils.pair_iterator(sorted(copy(data), key=lambda x: x[0])):
                new_X = b[0]-a[0]
                new_y = b[1]-a[1]
                new_data=(a[0]+(new_X/2), a[1]+(new_y/2))
                data.append(new_data)
                if len(data) == self.RANGE:
                    break

        while len(data) > self.RANGE:
            print("OOPS")
            data.pop()

        data=sorted(data, key=lambda x: x[0])


        x_train, y_train = zip(*data)
        x_train = np.array(x_train)
        y_train = np.array(y_train)
        x_train_transformed = np.array(
            [self.fourier_basis(x, self.fourier_degree) for x in x_train])
        return x_train, x_train_transformed, y_train, actualData_x, actualData_y

    def create_model(self, input_shape):
        model:tf.keras.Model = Sequential([
            Input(shape=input_shape),
            Dense(64, activation='relu'),
            Dense(1, activation='linear')
        ])
        model.compile(optimizer='adam', loss=tf.keras.losses.MeanSquaredError())
        model.summary()
        return model

    def train(self, test_data, queue=None, quiet=False):
        _, x_train_transformed, y_train, _, _ = self.prepared_data
        model=self.current_model

        _x = [self.fourier_basis(x, self.fourier_degree) for x in test_data]
        _x = np.array(_x)
        model.fit(x_train_transformed, y_train,
                       epochs=self.EPOCHS, callbacks=[MyCallback(queue, _x,quiet)], batch_size=32, verbose=0)
        self.current_model.save('./tmp/tmp_model.keras')
        


    def train_Process(self, test_data, queue=None, quiet=False)-> Process:
        # tf.keras.backend.clear_session()
        _, x_train_transformed, y_train, _, _ = self.prepared_data
        model=self.current_model

        _x = [self.fourier_basis(x, self.fourier_degree) for x in test_data]
        _x = np.array(_x)
        process=multiprocessing.Process(target=model.fit, args=(x_train_transformed, y_train), 
                                        kwargs={'epochs':self.EPOCHS, 'callbacks':[MyCallback(queue, _x,quiet)], 'batch_size':32, 'verbose':0})
        # 
        # process.start()
        return process

    def train_and_plot(self):
        x_train, x_train_transformed, y_train, actualData_x, actualData_y = self.prepared_data
        model= self.current_model

        plt.ion()
        fig, ax = plt.subplots()

        ax.clear()
        ax.scatter(actualData_x, actualData_y,
                   color='blue', label='Training data')
        _x, _y = zip(*list((x, self.fourier_basis(x, self.fourier_degree))
                     for x in np.linspace(-2 * np.pi, 2 * np.pi, self.RANGE)))
        y_test = model.predict(np.array(_y))
        ax.plot(_x, y_test, color='red', label='Neural network approximation')
        ax.legend()
        ax.grid()
        plt.draw()
        plt.pause(1)

        for epoch in range(self.ITTERRATIONS):
            print("Itterration ", epoch + 1, "/", self.ITTERRATIONS, sep="")
            self.model.fit(x_train_transformed, y_train, epochs=self.EPOCHS, batch_size=32, verbose=0)
        for epoch in range(self.ITTERRATIONS):
            print("epoch ", epoch + 1, "/", self.ITTERRATIONS, sep="")
            model.fit(x_train_transformed, y_train,
                           epochs=self.EPOCHS_PER_ITTERRATIONS, batch_size=32, verbose=0)

            ax.clear()
            ax.plot(_x, np.sin(_x), color='black', label='Base Frequency')
            ax.scatter(actualData_x, actualData_y,
                       color='blue', label='Training data')
            y_test = model.predict(np.array(_y))
            ax.plot(_x, y_test, '.-', color='red',
                    label='Neural network approximation', linewidth=2)
            ax.legend()
            ax.grid()
            plt.draw()
            try:
                plt.pause(0.1)
            except:
                pass
                

        plt.ioff()
        plt.show()

    def predict(self, data):
        if not hasattr(data, '__iter__'):
            data=[data]
        _y = list(self.fourier_basis(x, self.fourier_degree) for x in data)
        y_test = self.current_model.predict(np.array(_y))
        return y_test
    
    def synthesize(self, midi_notes, sample_rate=44100, duration=5.0):
        output = np.zeros(shape=int(sample_rate * duration))
        for note in midi_notes:
            freq = midi_to_freq(note)
            t = np.linspace(0, duration, int(sample_rate * duration), False)
            t_scaled = t * 2 * np.pi / (1/freq)
            signal = self.current_model.predict(np.array([self.fourier_basis(x,self.fourier_degree) for x in t_scaled]))
            output += signal
        output = output / np.max(np.abs(output))  # Normalize
        return output.astype(np.int16)
        
    def synthesize_2(self, midi_note, duration=1.0, sample_rate=44100):
        output = np.zeros(shape=int(sample_rate * duration))
        freq = midi_to_freq(midi_note)
        t = np.linspace(0, duration, int(sample_rate * duration), False)
        t_scaled = t * 2 * np.pi / (1/freq)
        output = self.current_model.predict(np.array([self.fourier_basis(x,self.fourier_degree) for x in t_scaled]))
        output = (output * 32767 / np.max(np.abs(output))) / 2  # Normalize
        return output.astype(np.int16)
    
    def synthesize_3(self, midi_note, duration=1.0, sample_rate=44100):
        sys.stdout = open(f'tmp/process_{os.getpid()}_output.txt', 'w', buffering=1)
        sys.stderr = open(f'tmp/process_{os.getpid()}_err.txt', 'w', buffering=1)
        print(f"begin generating sound for note: {midi_note}")
        output = np.zeros(shape=int(sample_rate * duration))
        freq = midi_to_freq(midi_note)
        t = np.linspace(0, duration, int(sample_rate * duration), False)
        t_scaled = t * 2 * np.pi / (1/freq)
        output = self.current_model.predict(np.array([self.fourier_basis(x,self.fourier_degree) for x in t_scaled]))#, batch_size=sample_rate/100)
        output = (output * 32767 / np.max(np.abs(output))) / 2  # Normalize
        print(f"Generated sound for note: {midi_note}")
        return (midi_note, output.astype(np.int16))
    


    def convert_to_audio(self, filename='./tmp/output.wav'):
        print(self.models.summary())
        sample_rate = self.RANGE
        duration = 5.0
        T = 1 / 440
        t = np.linspace(0, duration, int(sample_rate * duration), False)
        t_scaled = t * 2 * np.pi / T
        data=np.array([self.fourier_basis(x, self.fourier_degree) for x in t_scaled])
        print(data.shape)
        signal = self.models.predict(data)
        audio = (signal * 32767 / np.max(np.abs(signal))) / 2
        audio = audio.astype(np.int16)
        write(filename, sample_rate, audio)

    def save_model(self, filename='./tmp/model.h5'):
        self.current_model._name = os.path.basename(filename).replace('/', '').split('.')[0]
        self.current_model.save(filename)



    def load_new_model_from_file(self, filename='./tmp/model.h5'):
        if self.current_model:
            self.models.append(self.current_model)
        self.current_model = tf.keras.models.load_model(filename, )
        print(self.current_model.name)
        self.current_model._name = os.path.basename(filename).replace('/', '').split('.')[0]
        print(self.current_model.name)
        self.current_model.summary()
        print(self.current_model.input_shape)
        self.fourier_degree=self.current_model.input_shape[1]//2
        print(self.fourier_degree)

    def change_model(self, net_no):
        index=net_no-1
        if index>=0:
            new_model=self.models.pop(index)
            if self.current_model:
                self.models.insert(index, self.current_model)
            # self.fourier_degree=
            self.current_model=new_model




# moved to ./tests/fouriernn_test.py
# if __name__ == '__main__':
#     # Test the class with custom data points
#     data = [(x, np.sin(x * tf.keras.activations.relu(x)))
#             for x in np.linspace(np.pi, -np.pi, 100)]
#     fourier_nn = FourierNN(data)
#     fourier_nn.train()
# if __name__ == '__main__':
#     def relu(x):
#         return np.maximum(0, x)
#     # Test the class with custom data points
#     data = np.array([(x, np.sin(x * relu(x))) for x in np.linspace(np.pi, -np.pi, 100)])
#     fourier_nn = FourierNN(data)
#     fourier_nn.train_and_plot()
#     # fourier_nn.convert_to_audio()
#     # fourier_nn.save_model()