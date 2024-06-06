import os
os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"  # Set PlaidML as the backend for Keras

import random
import numpy as np
from keras.models import Sequential, load_model  # Import from keras instead of tensorflow.keras
from keras.layers import Dense  # Import from keras instead of tensorflow.keras
import matplotlib.pyplot as plt
from scipy.io.wavfile import write

class FourierNN():
    RANGE = 10**4
    ITTERRATIONS = 10
    EPOCHS = 10
    SIGNED_RANDOMNES = 0.00000001
    DEFAULT_FORIER_DEGREE = 100
    FORIER_DEGREE_DIVIDER = 100
    FORIER_DEGREE_OFFSET = 100

    def __init__(self, data):
        self.data = data
        self.model = None

    @staticmethod
    def fourier_basis(x, n=DEFAULT_FORIER_DEGREE):
        print("fourier_basis", x)
        basis = [np.sin(i * x) for i in range(1, n+1)]
        basis += [np.cos(i * x) for i in range(1, n+1)]
        return np.array(basis)

    @staticmethod
    def taylor_basis(x, n=DEFAULT_FORIER_DEGREE):
        basis = [x**i / np.math.factorial(i) for i in range(n)]
        return np.array(basis)

    def generate_data(self):
        print('generate_data')
        fourier_degree = (len(self.data) // self.FORIER_DEGREE_DIVIDER) + self.FORIER_DEGREE_OFFSET
        print('fourier_degree', fourier_degree)
        actualData_x, actualData_y = zip(*self.data)
        actualData_x = np.array(actualData_x)
        actualData_y = np.array(actualData_y)
        print(len(self.data), self.RANGE//len(self.data), self.RANGE)
        if len(self.data) < self.RANGE:
            print("extending data")
            ittr=0
            while len(self.data) < self.RANGE:
                print("itterration:", ittr, 'len', len(self.data), self.RANGE//len(self.data), self.RANGE)
                for i, data in enumerate(self.data):
                    print("element:", i, 'len', len(self.data), self.RANGE//len(self.data), self.RANGE)
                    if len(self.data) == self.RANGE:
                        break
                    new_data_X=(self.data[i-1][0]+data[0])/2
                    new_data_Y=(self.data[i-1][1]+data[1])/2
                    new_data=(self.data[i-1]+data)/2
                    self.data = np.insert(self.data, i, new_data, axis=0)
                ittr-=-1
        # if len(self.data) < self.RANGE:
        #     if len(self.data) == self.RANGE / 2:
        #         self.data.extend(self.data)
        #     else:
        #         multi = self.RANGE // len(self.data)
        #         rest = self.RANGE % len(self.data)
        #         self.data.extend([(dat[0] + random.uniform(-self.SIGNED_RANDOMNES, self.SIGNED_RANDOMNES),
        #                            dat[1] + random.uniform(-self.SIGNED_RANDOMNES, self.SIGNED_RANDOMNES)) for dat in self.data * multi])
        #         for i in range(rest):
        #             rand_x = random.uniform(-self.SIGNED_RANDOMNES, self.SIGNED_RANDOMNES)
        #             rand_y = random.uniform(-self.SIGNED_RANDOMNES, self.SIGNED_RANDOMNES)
        #             self.data.append((self.data[i][0] + rand_x, self.data[i][1] + rand_y))
        x_train, y_train = zip(*self.data)
        x_train = np.array(x_train)
        y_train = np.array(y_train)
        x_train_transformed = np.array([self.fourier_basis(x, fourier_degree) for x in x_train])
        return x_train_transformed, y_train, actualData_x, actualData_y, fourier_degree

    def create_model(self, input_shape):
        self.model = Sequential([
            Dense(64,input_shape=input_shape),
            Dense(64, activation='relu'),
            Dense(1)
        ])
        self.model.compile(optimizer='adam', loss='mean_squared_error')

    def train_and_plot(self):
        print('train_and_plot')
        x_train_transformed, y_train, actualData_x, actualData_y, fourier_degree = self.generate_data()
        self.create_model((x_train_transformed.shape[1],))
        print('uff')

        plt.ion()
        fig, ax = plt.subplots()

        ax.clear()
        ax.scatter(actualData_x, actualData_y, color='blue', label='Training data')
        _x, _y = zip(*list((x, self.fourier_basis(x, fourier_degree)) for x in np.linspace(-2 * np.pi, 2 * np.pi, self.RANGE)))
        y_test = self.model.predict(np.array(_y))
        ax.plot(_x, y_test, color='red', label='Neural network approximation')
        ax.legend()
        ax.grid()
        plt.draw()
        plt.pause(1)

        for epoch in range(self.ITTERRATIONS):
            print("Itterration ", epoch + 1, "/", self.ITTERRATIONS, sep="")
            self.model.fit(x_train_transformed, y_train, epochs=self.EPOCHS, batch_size=32, verbose=1)

            ax.clear()
            ax.plot(_x, np.sin(_x), color='black', label='Base Frequency')
            ax.scatter(actualData_x, actualData_y, color='blue', label='Training data')
            y_test = self.model.predict(np.array(_y))
            ax.plot(_x, y_test, '.-', color='red', label='Neural network approximation', linewidth=2)
            ax.legend()
            ax.grid()
            plt.draw()
            plt.pause(0.1)

        plt.ioff()
        plt.show()

    def convert_to_audio(self, filename='output.wav'):
        sample_rate = 44100
        duration = 5.0
        T = 1 / 440
        t = np.linspace(0, duration, int(sample_rate * duration), False)
        t_scaled = t * 2 * np.pi / T

        signal = self.model.predict(np.array([self.fourier_basis(x) for x in t_scaled]))
        audio = (signal * 32767 / np.max(np.abs(signal))) / 2
        audio = audio.astype(np.int16)
        write(filename, sample_rate, audio)

    def save_model(self, filename='model.h5'):
        self.model.save(filename)

    def load_model(self, filename='model.h5'):
        self.model = load_model(filename)

if __name__ == '__main__':
    def relu(x):
        return np.maximum(0, x)
    # Test the class with custom data points
    data = np.array([(x, np.sin(x * relu(x))) for x in np.linspace(np.pi, -np.pi, 100)])
    fourier_nn = FourierNN(data)
    fourier_nn.train_and_plot()
    # fourier_nn.convert_to_audio()
    # fourier_nn.save_model()
