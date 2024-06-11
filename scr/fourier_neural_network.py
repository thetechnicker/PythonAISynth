import random
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Input
from keras.callbacks import Callback
import matplotlib.pyplot as plt
from scipy.io.wavfile import write
from multiprocessing import Process, Queue

from scr.utils import midi_to_freq

class MyCallback(Callback):
    def __init__(self, queue:Queue=None, test_data=None, quiet=False):
        super().__init__()
        self.queue:Queue = queue
        self.test_data = test_data
        self.quiet = quiet

    def on_epoch_begin(self, epoch, logs=None):
        if not self.quiet:
            print(f"EPOCHE {epoch} STARTED")

    def on_epoch_end(self, epoch, logs=None):
        if self.queue:
            self.queue.put(self.test_data[0], self.model.predict(self.test_data[1]))
        if not self.quiet:
            print(f"EPOCHE {epoch} ENDED, Logs: {logs}")


class FourierNN:
    RANGE = 44100

    ITTERRATIONS = 5
    EPOCHS_PER_ITTERRATIONS = 1

    EPOCHS = 100

    SIGNED_RANDOMNES = 0.000000001
    DEFAULT_FORIER_DEGREE = 10
    FORIER_DEGREE_DIVIDER = 100
    FORIER_DEGREE_OFFSET = 1

    def __init__(self, data):
        self.data = data
        self.model = None
        self.fourier_degree=self.DEFAULT_FORIER_DEGREE

    @staticmethod
    def fourier_basis(x, n=DEFAULT_FORIER_DEGREE):
        basis = [np.sin(i * x) for i in range(1, n+1)]
        basis += [np.cos(i * x) for i in range(1, n+1)]
        return np.array(basis)

    @staticmethod
    def taylor_basis(x, n=DEFAULT_FORIER_DEGREE):
        basis = [x**i / np.math.factorial(i) for i in range(n)]
        return np.array(basis)

    def generate_data(self):
        self.fourier_degree = (
            len(self.data) // self.FORIER_DEGREE_DIVIDER) + self.FORIER_DEGREE_OFFSET
        actualData_x, actualData_y = zip(*self.data)
        if len(self.data) < self.RANGE:
            if len(self.data) == self.RANGE / 2:
                self.data.extend(self.data)
            else:
                multi = self.RANGE // len(self.data)
                rest = self.RANGE % len(self.data)
                self.data.extend([(dat[0] + random.uniform(-self.SIGNED_RANDOMNES, self.SIGNED_RANDOMNES),
                                   dat[1] + random.uniform(-self.SIGNED_RANDOMNES, self.SIGNED_RANDOMNES)) for dat in self.data * multi])
                for i in range(rest):
                    rand_x = random.uniform(-self.SIGNED_RANDOMNES,
                                            self.SIGNED_RANDOMNES)
                    rand_y = random.uniform(-self.SIGNED_RANDOMNES,
                                            self.SIGNED_RANDOMNES)
                    self.data.append(
                        (self.data[i][0] + rand_x, self.data[i][1] + rand_y))
        x_train, y_train = zip(*self.data)
        x_train = np.array(x_train)
        y_train = np.array(y_train)
        x_train_transformed = np.array(
            [self.fourier_basis(x, self.fourier_degree) for x in x_train])
        return x_train, x_train_transformed, y_train, actualData_x, actualData_y, self.fourier_degree

    def create_model(self, input_shape):
        self.model:Sequential = Sequential([
            Input(shape=input_shape),
            Dense(64, activation='relu'),
            Dense(1)
        ])
        self.model.compile(optimizer='adam', loss='mean_squared_error')
        print(self.model.summary())

    def train(self, queue=None, quiet=False):
        _, x_train_transformed, y_train, _, _, fourier_degree = self.generate_data()
        self.create_model((x_train_transformed.shape[1],))

        _x, _y = zip(*list((x, self.fourier_basis(x, fourier_degree))
                     for x in np.linspace(-2 * np.pi, 2 * np.pi, self.RANGE)))
        _x, _y = np.array(_x), np.array(_y)
        self.model.fit(x_train_transformed, y_train,
                       epochs=self.EPOCHS, callbacks=[MyCallback(queue, (_x, _y,),quiet)], batch_size=32, verbose=0)

    def train_and_plot(self):
        x_train, x_train_transformed, y_train, actualData_x, actualData_y, fourier_degree = self.generate_data()
        self.create_model((x_train_transformed.shape[1],))

        plt.ion()
        fig, ax = plt.subplots()

        ax.clear()
        ax.scatter(actualData_x, actualData_y,
                   color='blue', label='Training data')
        _x, _y = zip(*list((x, self.fourier_basis(x, fourier_degree))
                     for x in np.linspace(-2 * np.pi, 2 * np.pi, self.RANGE)))
        y_test = self.model.predict(np.array(_y))
        ax.plot(_x, y_test, color='red', label='Neural network approximation')
        ax.legend()
        ax.grid()
        plt.draw()
        plt.pause(1)

        for epoch in range(self.ITTERRATIONS):
            print("epoch ", epoch + 1, "/", self.ITTERRATIONS, sep="")
            self.model.fit(x_train_transformed, y_train,
                           epochs=self.EPOCHS_PER_ITTERRATIONS, batch_size=32, verbose=0)

            ax.clear()
            ax.plot(_x, np.sin(_x), color='black', label='Base Frequency')
            ax.scatter(actualData_x, actualData_y,
                       color='blue', label='Training data')
            y_test = self.model.predict(np.array(_y))
            ax.plot(_x, y_test, '.-', color='red',
                    label='Neural network approximation', linewidth=2)
            ax.legend()
            ax.grid()
            plt.draw()
            plt.pause(0.1)

        plt.ioff()
        plt.show()

    def predict(self, data):
        _, _y = zip(*list((x, self.fourier_basis(x, self.fourier_degree))
                     for x in data))
        y_test = self.model.predict(np.array(_y))
        return y_test


    def synthesize(self, midi_notes, model, sample_rate=44100, duration=5.0):
        output = np.zeros(int(sample_rate * duration))
        for note in midi_notes:
            freq = midi_to_freq(note)
            t = np.linspace(0, duration, int(sample_rate * duration), False)
            t_scaled = t * 2 * np.pi / freq
            signal = self.model.predict(np.array([self.fourier_basis(x,self.fourier_degree) for x in t_scaled]))
            output += signal
        output = output / np.max(np.abs(output))  # Normalize
        return output.astype(np.int16)
    

    def convert_to_audio(self, filename='output.wav'):
        print(self.model.summary())
        sample_rate = self.RANGE
        duration = 5.0
        T = 1 / 440
        t = np.linspace(0, duration, int(sample_rate * duration), False)
        t_scaled = t * 2 * np.pi / T
        data=np.array([self.fourier_basis(x, self.fourier_degree) for x in t_scaled])
        print(data.shape)
        signal = self.model.predict(data)
        audio = (signal * 32767 / np.max(np.abs(signal))) / 2
        audio = audio.astype(np.int16)
        write(filename, sample_rate, audio)

    def save_model(self, filename='model.h5'):
        self.model.save(filename)

    def load_model(self, filename='model.h5'):
        self.model = tf.keras.models.load_model(filename)

# moved to ./tests/fouriernn_test.py
# if __name__ == '__main__':
#     # Test the class with custom data points
#     data = [(x, np.sin(x * tf.keras.activations.relu(x)))
#             for x in np.linspace(np.pi, -np.pi, 100)]
#     fourier_nn = FourierNN(data)
#     fourier_nn.train()
