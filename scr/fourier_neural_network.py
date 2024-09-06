import os
import sys
import multiprocessing
from multiprocessing.pool import Pool
import time
from typing import Tuple
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Input
from keras.callbacks import Callback
from tensorflow.keras.callbacks import EarlyStopping, CSVLogger
from multiprocessing import Queue, current_process
from concurrent.futures import ProcessPoolExecutor
from numba import njit, prange

from scr import utils
from scr.utils import midi_to_freq
# print(tf.__version__)


class MyCallback(Callback):
    def __init__(self, queue: Queue = None, test_data=None, quiet=False):
        # print(tf.__version__)
        super().__init__()
        self.queue: Queue = queue
        self.test_data = test_data
        self.quiet = quiet

    def on_epoch_begin(self, epoch, logs=None):
        self.timestamp = time.perf_counter_ns()
        if not self.quiet:
            print(f"EPOCHE {epoch+1} STARTED")

    def on_epoch_end(self, epoch, logs=None):
        if self.queue:
            self.queue.put(self.model.predict(
                self.test_data, batch_size=len(self.test_data)))
        if not self.quiet:
            time_taken = time.perf_counter_ns()-self.timestamp
            print(
                f"EPOCHE {epoch+1} ENDED, loss: {logs['loss']}, val_loss: {logs['val_loss']}. Time Taken: {time_taken/1_000_000_000}s")


class FourierNN:
    SAMPLES = 44100//2
    EPOCHS = 100
    CALC_FOURIER_DEGREE_BY_DATA_LENGTH = False
    DEFAULT_FORIER_DEGREE = 250
    FORIER_DEGREE_DIVIDER = 1
    FORIER_DEGREE_OFFSET = 0
    PATIENCE = 10
    OPTIMIZER = 'Adam'
    LOSS_FUNCTION = 'Huber'
    change_params = False
    keys = [
        'SAMPLES',
        'EPOCHS',
        'CALC_FOURIER_DEGREE_BY_DATA_LENGTH',
        'DEFAULT_FORIER_DEGREE',
        'FORIER_DEGREE_DIVIDER',
        'FORIER_DEGREE_OFFSET',
        'PATIENCE',
        'OPTIMIZER',
        'LOSS_FUNCTION',
    ]

    def update_attribs(self, **kwargs):
        for key, val in kwargs.items():
            if hasattr(self, key) and key in self.keys:
                setattr(self, key, val)
                change_params = True
                # if self.prepared_data:
                #    self.create_new_model()
            else:
                raise Exception(
                    f"The attribute '{key}' does not exist or can not be modified this way.")
        for key in self.keys:
            val = getattr(self, key, None)
            print(f"{key}: {val}")
        self.create_new_model()
        print("-------------------------")

    def __getstate__(self):
        # Save the model to a file before serialization
        # self.current_model.save('./tmp/tmp_model.keras')
        model = self.current_model
        del self.current_model
        fourier_nn_dict = self.__dict__.copy()
        self.current_model = model
        return fourier_nn_dict

    def __setstate__(self, state):
        # Load the model from a file after deserialization
        self.__dict__.update(state)
        if os.path.exists('./tmp/tmp_model.keras'):
            with self.lock:
                self.current_model = tf.keras.models.load_model(
                    './tmp/tmp_model.keras')

        if current_process().name != 'MainProcess' and self.stdout_queue:
            sys.stdout = utils.QueueSTD_OUT(self.stdout_queue)
        # for key in self.keys:
        #     val = getattr(self, key, None)
        #     print(f"{key}: {val}")
        # self.stdout_queue = None
        # print("-------------------------")

    def load_tmp_model(self):
        with self.lock:
            self.current_model = tf.keras.models.load_model(
                './tmp/tmp_model.keras')
            self.current_model.summary()

    def save_tmp_model(self):
        self.current_model.save('./tmp/tmp_model.keras')

    def __init__(self, lock, data=None, stdout_queue=None):
        self.lock = lock
        self.models: list = []
        self.current_model: keras.Model = None
        self.fourier_degree = self.DEFAULT_FORIER_DEGREE
        self.prepared_data = None
        self.stdout_queue = stdout_queue
        if data is not None:
            self.update_data(data)

    def create_new_model(self):
        if hasattr(self, "prepared_data"):
            if self.prepared_data:
                # if self.current_model:
                #     self.models.append(self.current_model)
                self.current_model = self.create_model(
                    (self.prepared_data[1].shape[1],))

    def get_models(self) -> list[Sequential]:
        return [self.current_model]+self.models

    def update_data(self, data, stdout_queue=None):
        self.stdout_queue = stdout_queue
        self.prepared_data = self.prepare_data(list(data))
        if not self.current_model:
            self.create_new_model()

    def get_trainings_data(self):
        return list(zip(self.prepared_data[0], self.prepared_data[2]))

    @staticmethod
    @njit(parallel=True)
    def fourier_basis_numba(data, indices):
        n_samples = data.shape[0]
        n_features = indices.size * 2
        result = np.empty((n_samples, n_features), dtype=np.float64)

        for i in prange(n_samples):
            x = data[i]
            print("generating fourier basis for:", x)
            sin_basis = np.sin(np.outer(indices, x))
            cos_basis = np.cos(np.outer(indices, x))
            basis = np.concatenate((sin_basis, cos_basis), axis=0)
            result[i, :] = basis.flatten()

        return result

    @staticmethod
    def precompute_indices(n=DEFAULT_FORIER_DEGREE):
        return np.arange(1, n + 1)

    @staticmethod
    def fourier_basis(x, indices):
        sin_basis = np.sin(np.outer(indices, x))
        cos_basis = np.cos(np.outer(indices, x))
        basis = np.concatenate((sin_basis, cos_basis), axis=0).flatten()
        return basis

    @staticmethod
    def taylor_basis(x, n=DEFAULT_FORIER_DEGREE):
        basis = [x**i / np.math.factorial(i) for i in range(n)]
        return np.array(basis)

    def prepare_data(self, data):

        self.fourier_degree = ((len(data) // self.FORIER_DEGREE_DIVIDER) + self.FORIER_DEGREE_OFFSET
                               if self.CALC_FOURIER_DEGREE_BY_DATA_LENGTH
                               else self.DEFAULT_FORIER_DEGREE)

        actualData_x, actualData_y = zip(*data)

        # "math" data filling
        data = sorted(data, key=lambda x: x[0])
        while len(data) < self.SAMPLES:
            data_copy = sorted(data[:], key=lambda x: x[0])
            pairs = ((data_copy[i], data_copy[i+1], 0.5)
                     for i in range(len(data_copy)-1))
            with Pool() as pool:
                new_data = pool.starmap(utils.interpolate, pairs)
            data.extend(new_data)
            if len(data) >= self.SAMPLES:
                # Delete every second element
                data = data[::2]
                break

        while len(data) > self.SAMPLES:
            data.pop()

        data = sorted(data, key=lambda x: x[0])

        # gen test_data
        test_data = []
        test_samples = self.SAMPLES/2
        # "math" data filling
        test_data = sorted(data, key=lambda x: x[0])
        while len(test_data) < test_samples:
            data_copy = sorted(test_data[:], key=lambda x: x[0])
            pairs = ((data_copy[i], data_copy[i+1],)
                     for i in range(len(data_copy)-1))
            with Pool() as pool:
                new_data = pool.starmap(utils.interpolate, pairs)
            test_data.extend(new_data)
            if len(test_data) >= test_samples:
                # Delete every second element
                test_data = test_data[::2]
                break

        while len(test_data) > test_samples:
            test_data.pop()

        test_data = sorted(test_data, key=lambda x: x[0])

        x_train, y_train = zip(*data)
        x_test, y_test = zip(*test_data)

        x_train = np.array(x_train)
        y_train = np.array(y_train)
        x_test = np.array(x_test)
        y_test = np.array(y_test)
        print(" Generate Fourier-Basis ".center(50, '-'))
        indices = FourierNN.precompute_indices(self.fourier_degree)
        # with multiprocessing.Pool(processes=os.cpu_count()) as pool:
        #     result_a = pool.starmap(FourierNN.fourier_basis, zip(
        #         x_train,
        #         (indices for i in range(len(x_train))),
        #     ))
        #     result_b = pool.starmap(FourierNN.fourier_basis, zip(
        #         x_test,
        #         (indices for i in range(len(x_test))),
        #     ))

        result_a = self.fourier_basis_numba(x_train, indices)
        result_b = self.fourier_basis_numba(x_train, indices)

        print(''.center(50, '-'))

        # .reshape(-1, self.fourier_degree*2)
        x_train_transformed = np.array(result_a)
        x_test_transformed = np.array(result_b)
        print(len(x_train_transformed), len(y_train))
        return (x_train, x_train_transformed, y_train, actualData_x, actualData_y, x_test_transformed, y_test)

    def create_model(self, input_shape):
        print(input_shape)
        model: tf.keras.Model = Sequential([
            Input(shape=input_shape),
            Dense(64, activation='linear'),
            Dense(1, activation='linear')
        ])
        model.compile(optimizer=self.OPTIMIZER,
                      loss=tf.keras.losses.get(self.LOSS_FUNCTION),
                      metrics=[
                          'accuracy',
                          tf.keras.metrics.MeanSquaredError(),
                          tf.keras.metrics.MeanAbsoluteError(),
                          tf.keras.metrics.RootMeanSquaredError(),
                      ])
        model.summary()
        return model

    def train(self, test_data, queue=None, quiet=False):
        _, x_train_transformed, y_train, _, _, test_x, test_y = self.prepared_data
        if self.change_params:
            self.create_new_model()
        model = self.current_model
        indices = FourierNN.precompute_indices(self.fourier_degree)
        _x = [self.fourier_basis(x, indices) for x in test_data]
        _x = np.array(_x)
        early_stopping = EarlyStopping(
            monitor='val_loss', patience=self.PATIENCE)
        csvlogger = CSVLogger("./tmp/training.csv")

        print(model.input_shape, x_train_transformed.shape, y_train.shape)
        model.fit(x_train_transformed, y_train,
                  epochs=self.EPOCHS, validation_data=(test_x, test_y),
                  callbacks=[
                      MyCallback(queue, _x, quiet),
                      early_stopping,
                      csvlogger,
                  ],
                  batch_size=int(self.SAMPLES/2), verbose=0)
        self.current_model.save('./tmp/tmp_model.keras')

    def predict(self, data, batch_size=None):
        indices = FourierNN.precompute_indices(self.fourier_degree)
        if not hasattr(data, '__iter__'):
            _y = [FourierNN.fourier_basis(data, indices)]
        else:
            data = np.array(data)

            def t():
                return [FourierNN.fourier_basis(x, indices) for x in data]
            # _y = utils.messure_time_taken(
            #     "fourier_basis", FourierNN.fourier_basis_numba, data, indices, wait=False)
            _y = utils.messure_time_taken(
                "fourier_basis", t, wait=False)
        _y = np.array(_y)
        print(_y.shape)
        _y = _y.reshape(len(_y), -1)
        y_test = self.current_model.predict(
            np.array(_y), batch_size=batch_size if batch_size else len(data))

        return y_test

    def synthesize(self, midi_note, duration=1.0, sample_rate=44100):
        output = np.zeros(shape=int(sample_rate * duration))
        freq = midi_to_freq(midi_note)
        t = np.linspace(0, duration, int(sample_rate * duration), False)
        t_scaled = t * 2 * np.pi / (1/freq)
        batch_size = int(sample_rate*duration)
        output = self.current_model.predict(np.array([self.fourier_basis(
            x, self.fourier_degree) for x in t_scaled]), batch_size=batch_size)
        output = (output * 32767 / np.max(np.abs(output))) / 2  # Normalize
        return output.astype(np.int16)

    def synthesize_tuple(self, midi_note, duration=1.0, sample_rate=44100) -> Tuple[int, np.array]:
        print(f"begin generating sound for note: {midi_note}", flush=True)
        output = np.zeros(shape=int(sample_rate * duration))
        freq = midi_to_freq(midi_note)
        t = np.linspace(0, duration, int(sample_rate * duration), False)
        t_scaled = (t * 2 * np.pi) / (1/freq)
        batch_size = sample_rate//1
        output = self.current_model.predict(np.array([self.fourier_basis(
            x, self.fourier_degree) for x in t_scaled]), batch_size=batch_size)
        output = (output * 32767 / np.max(np.abs(output))) / 2  # Normalize
        print(f"Generated sound for note: {midi_note}")
        # gc.collect()
        return (midi_note, output.astype(np.int16))

    def save_model(self, filename='./tmp/model.h5'):
        self.current_model._name = os.path.basename(
            filename).replace('/', '').split('.')[0]
        self.current_model.save(filename)

    def load_new_model_from_file(self, filename='./tmp/model.h5'):
        if self.current_model:
            self.models.append(self.current_model)
        self.current_model = tf.keras.models.load_model(filename)
        print(self.current_model.name)
        self.current_model._name = os.path.basename(
            filename).replace('/', '').split('.')[0]
        print(self.current_model.name)
        self.current_model.summary()
        print(self.current_model.input_shape)
        self.fourier_degree = self.current_model.input_shape[1]//2
        print(self.fourier_degree)

    def change_model(self, net_no):
        index = net_no-1
        if index >= 0:
            new_model = self.models.pop(index)
            if self.current_model:
                self.models.insert(index, self.current_model)
            # self.fourier_degree=
            self.current_model = new_model
