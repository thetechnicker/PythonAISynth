import random
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
import matplotlib.pyplot as plt
from scipy.io.wavfile import write

class FourierNN:
    RANGE = 10**3
    EPOCHS = 50
    TRAINING_SPEED = 1
    SIGNED_RANDOMNES = 0.000000001
    DEFAULT_FORIER_DEGREE = 10
    FORIER_DEGREE_DIVIDER = 100
    FORIER_DEGREE_OFFSET = 1

    def __init__(self, data):
        self.data = data
        self.model = None

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
        fourier_degree = (len(self.data) // self.FORIER_DEGREE_DIVIDER) + self.FORIER_DEGREE_OFFSET
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
                    rand_x = random.uniform(-self.SIGNED_RANDOMNES, self.SIGNED_RANDOMNES)
                    rand_y = random.uniform(-self.SIGNED_RANDOMNES, self.SIGNED_RANDOMNES)
                    self.data.append((self.data[i][0] + rand_x, self.data[i][1] + rand_y))
        x_train, y_train = zip(*self.data)
        x_train = np.array(x_train)
        y_train = np.array(y_train)
        x_train_transformed = np.array([self.fourier_basis(x, fourier_degree) for x in x_train])
        return x_train, x_train_transformed, y_train, actualData_x, actualData_y, fourier_degree

    def create_model(self, input_shape):
        self.model = Sequential([
            Input(shape=input_shape),
            Dense(64, activation='relu'),
            Dense(1)
        ])
        self.model.compile(optimizer='adam', loss='mean_squared_error')

    def train_and_plot(self):
        x_train, x_train_transformed, y_train, actualData_x, actualData_y, fourier_degree = self.generate_data()
        self.create_model((x_train_transformed.shape[1],))

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

        for epoch in range(self.EPOCHS):
            print("epoch ", epoch + 1, "/", self.EPOCHS, sep="")
            self.model.fit(x_train_transformed, y_train, epochs=self.TRAINING_SPEED, batch_size=32, verbose=0)

            ax.clear()
            ax.plot(_x, np.sin(_x), color='black', label='Base Frequency')
            ax.scatter(actualData_x, actualData_y, color='blue', label='Training data')
            y_test = self.model.predict(np.array(_y))
            ax.plot(_x, y_test, 'o-', color='red', label='Neural network approximation', linewidth=2)
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
        self.model = tf.keras.models.load_model(filename)

if __name__ == '__main__':
    # Test the class with custom data points
    data = [(x, np.sin(x * tf.keras.activations.relu(x))) for x in np.linspace(np.pi, -np.pi, 100)]
    fourier_nn = FourierNN(data)
    fourier_nn.train_and_plot()
    # fourier_nn.convert_to_audio()
    # fourier_nn.save_model()
