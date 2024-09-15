import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from typing import Tuple
from numba import njit, prange
from multiprocessing import Queue, Pool
from numba import njit, prange

from scr import utils
from scr.utils import QueueSTD_OUT, midi_to_freq

try:
    import torch_directml
    DIRECTML = True
except ImportError:
    DIRECTML = False


class MyCallback:
    def __init__(self, queue: Queue = None, test_data=None, quiet=False, std_queue: Queue = None):
        self.queue = queue
        self.test_data = test_data
        self.quiet = quiet

    def on_epoch_begin(self, epoch):
        self.timestamp = time.perf_counter_ns()
        if not self.quiet:
            print(f"EPOCH {epoch + 1} STARTED")

    def on_epoch_end(self, epoch, loss, val_loss, model):
        if self.queue:
            with torch.no_grad():
                predictions = model(self.test_data)
                self.queue.put(predictions.cpu().numpy())
        if not self.quiet:
            self.timestamp = time.perf_counter_ns()
            time_taken = time.perf_counter_ns() - self.timestamp
            print(
                f"EPOCH {epoch + 1} ENDED, loss: {loss}, val_loss: {val_loss}. Time Taken: {time_taken / 1_000_000_000}s")


class FourierNN(nn.Module):
    SAMPLES = 1000
    EPOCHS = 1000
    DEFAULT_FORIER_DEGREE = 300
    FORIER_DEGREE_DIVIDER = 1
    FORIER_DEGREE_OFFSET = 0
    PATIENCE = 50
    OPTIMIZER = 'Adam'
    LOSS_FUNCTION = nn.HuberLoss()
    CALC_FOURIER_DEGREE_BY_DATA_LENGTH = False

    def __init__(self, lock, data=None, stdout_queue=None):
        super(FourierNN, self).__init__()
        self.lock = lock
        self.models = []
        self.current_model = None
        self.fourier_degree = self.DEFAULT_FORIER_DEGREE
        self.prepared_data = None
        self.stdout_queue = stdout_queue
        if data is not None:
            self.update_data(data)

        if DIRECTML:
            self.device = torch_directml.device() if torch_directml.is_available() else None
        if not self.device:
            if torch.cuda.is_available() and torch.version.hip:
                self.device = torch.device('rocm')
            elif torch.cuda.is_available():
                self.device = torch.device('cuda')
            else:
                self.device = torch.device('cpu')
        print(self.device)

    def update_attribs(self, **kwargs):
        """
        Update model parameters.
        Accepts keyword arguments for parameters to update.
        """
        for key, val in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, val)
            else:
                raise ValueError(
                    f"Parameter '{key}' does not exist in the model.")

        # If the data has been prepared, recreate the model with updated parameters
        if self.prepared_data:
            self.create_new_model()

    def create_model(self, input_shape):
        model = nn.Sequential(
            nn.Linear(input_shape[0], 1),
            # nn.ReLU(),
            # nn.Linear(64, 1)
        )
        return model

    def update_data(self, data, stdout_queue=None):
        if stdout_queue:
            self.stdout_queue = stdout_queue
        self.prepared_data = self.prepare_data(
            list(data), std_queue=self.stdout_queue)
        if not self.current_model:
            self.current_model = self.create_model(
                (self.prepared_data[0].shape[1],))

    def prepare_data(self, data, std_queue: Queue = None):
        if std_queue:
            self.stdout_queue = std_queue

        self.fourier_degree = (
            (len(data) // self.FORIER_DEGREE_DIVIDER) + self.FORIER_DEGREE_OFFSET)

        # "math" data filling
        data = sorted(data, key=lambda x: x[0])
        while len(data) < self.SAMPLES:
            data_copy = sorted(data[:], key=lambda x: x[0])
            pairs = ((data_copy[i], data_copy[i + 1], 0.5)
                     for i in range(len(data_copy) - 1))
            with Pool(processes=os.cpu_count()) as pool:
                new_data = pool.starmap(utils.interpolate, pairs)
            data.extend(new_data)
            if len(data) >= self.SAMPLES:
                break

        while len(data) > self.SAMPLES:
            data.pop()

        data = sorted(data, key=lambda x: x[0])

        # Generate test_data
        test_data = []
        test_samples = self.SAMPLES / 2
        test_data = sorted(data, key=lambda x: x[0])
        while len(test_data) < test_samples:
            data_copy = sorted(test_data[:], key=lambda x: x[0])
            pairs = ((data_copy[i], data_copy[i + 1],)
                     for i in range(len(data_copy) - 1))
            with Pool(processes=os.cpu_count()) as pool:
                new_data = pool.starmap(utils.interpolate, pairs)
            test_data.extend(new_data)
            if len(test_data) >= test_samples:
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

        indices = FourierNN.precompute_indices(self.fourier_degree)

        result_a = self.fourier_basis_numba(x_train, indices)
        result_b = self.fourier_basis_numba(x_test, indices)

        x_train_transformed = np.array(result_a)
        x_test_transformed = np.array(result_b)

        return (torch.tensor(x_train_transformed,
                             dtype=torch.float32),
                torch.tensor(y_train,
                             dtype=torch.float32),
                torch.tensor(x_test_transformed,
                             dtype=torch.float32),
                torch.tensor(y_test,
                             dtype=torch.float32))

    @staticmethod
    @njit(parallel=True)
    def fourier_basis_numba(data, indices):
        n_samples = data.shape[0]
        n_features = indices.size * 2
        result = np.empty((n_samples, n_features), dtype=np.float64)

        for i in prange(n_samples):
            x = data[i]
            sin_basis = np.sin(np.outer(indices, x))
            cos_basis = np.cos(np.outer(indices, x))
            basis = np.concatenate((sin_basis, cos_basis), axis=0)
            result[i, :] = basis.flatten()

        return result

    @staticmethod
    def precompute_indices(n=DEFAULT_FORIER_DEGREE):
        return np.arange(1, n + 1)

    def train(self, test_data, queue=None, quiet=False, stdout_queue=None):
        print(self.OPTIMIZER,
              self.LOSS_FUNCTION,
              sep="\n")
        # exit(-1)
        if stdout_queue:
            sys.stdout = QueueSTD_OUT(stdout_queue)

        x_train_transformed, y_train, test_x, test_y = self.prepared_data

        model = self.current_model.to(self.device)

        x_train_transformed.to(self.device)
        y_train.to(self.device)
        test_x.to(self.device)
        test_y.to(self.device)

        optimizer = utils.get_optimizer(
            optimizer_name=self.OPTIMIZER,
            model_parameters=model.parameters(),
            lr=0.001)
        criterion = utils.get_loss_function(self.LOSS_FUNCTION)

        train_dataset = TensorDataset(x_train_transformed, y_train)
        train_loader = DataLoader(
            train_dataset, batch_size=int(self.SAMPLES / 2), shuffle=True)

        prepared_test_data = torch.tensor(
            data=FourierNN.fourier_basis_numba(
                data=test_data.flatten(),
                indices=FourierNN.precompute_indices(self.fourier_degree)),
            dtype=torch.float32).to(self.device)

        # prepared_test_data.to(self.device)

        # callback = MyCallback(
        #     queue=queue, test_data=prepared_test_data, quiet=quiet)

        for epoch in range(self.EPOCHS):
            # callback.on_epoch_begin(epoch)
            print(f"epoch {epoch+1} beginns")
            timestamp = time.perf_counter_ns()
            model.train()
            epoch_loss = 0
            for batch_x, batch_y in train_loader:
                batch_x, batch_y = batch_x.to(
                    self.device), batch_y.to(self.device)
                optimizer.zero_grad()
                outputs = model(batch_x)
                loss = criterion(outputs.squeeze(), batch_y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            epoch_loss /= len(train_loader)

            # Validation
            model.eval()
            with torch.no_grad():
                val_outputs = model(test_x.to(self.device))
                val_loss = criterion(val_outputs.squeeze(),
                                     test_y.to(self.device))
                if queue:
                    predictions = model(prepared_test_data)
                    queue.put(predictions.cpu().numpy())

            # callback.on_epoch_end(epoch, epoch_loss, val_loss.item(), model)
            time_taken = time.perf_counter_ns() - timestamp
            print(f"epoch {epoch+1} ends. "
                  f"loss:{epoch_loss}, "
                  f"val_loss: {val_loss}. "
                  f"Time Taken: {time_taken / 1_000_000_000}s")

        torch.save(model.state_dict(), './tmp/tmp_model.pth')

    def predict(self, data):
        indices = FourierNN.precompute_indices(self.fourier_degree)
        if not hasattr(data, '__iter__'):
            x = torch.tensor(
                [self.fourier_basis(data, indices)], dtype=torch.float32)
        else:
            data = np.array(data)
            x = self.fourier_basis_numba(data, indices)
            x = torch.tensor(x, dtype=torch.float32)

        with torch.no_grad():
            y = self.current_model(x)

        return y.numpy()

    def save_model(self, filename='./tmp/model.pth'):
        torch.save(self.current_model.state_dict(), filename)

    def load_new_model_from_file(self, filename='./tmp/model.pth'):
        self.current_model = self.create_model(
            (self.prepared_data[0].shape[1],))
        self.current_model.load_state_dict(torch.load(filename))
        self.current_model.eval()
        print(self.current_model)

    def create_new_model(self):
        if hasattr(self, "prepared_data"):
            if self.prepared_data:
                self.current_model = self.create_model(
                    (self.prepared_data[0].shape[1],))

    def change_model(self, net_no):
        index = net_no - 1
        if index >= 0:
            new_model = self.models.pop(index)
            if self.current_model:
                self.models.insert(index, self.current_model)
            self.current_model = new_model

    def synthesize(self, midi_note, duration=1.0, sample_rate=44100):
        output = np.zeros(shape=int(sample_rate * duration))
        freq = midi_to_freq(midi_note)
        t = np.linspace(0, duration, int(sample_rate * duration), False)
        t_scaled = t * 2 * np.pi / (1 / freq)
        batch_size = int(sample_rate * duration)
        x = torch.tensor([self.fourier_basis(x, self.fourier_degree)
                         for x in t_scaled], dtype=torch.float32)
        output = self.current_model(x).numpy()
        output = (output * 32767 / np.max(np.abs(output))) / 2  # Normalize
        return output.astype(np.int16)

    def synthesize_tuple(self, midi_note, duration=1.0, sample_rate=44100) -> Tuple[int, np.array]:
        print(f"begin generating sound for note: {midi_note}", flush=True)
        output = np.zeros(shape=int(sample_rate * duration))
        freq = midi_to_freq(midi_note)
        t = np.linspace(0, duration, int(sample_rate * duration), False)
        t_scaled = (t * 2 * np.pi) / (1 / freq)
        batch_size = sample_rate // 1
        indices = FourierNN.precompute_indices(self.fourier_degree)
        x = self.fourier_basis_numba(t_scaled, indices)
        x = torch.tensor(x, dtype=torch.float32)
        output = self.current_model(x).numpy()
        output = (output * 32767 / np.max(np.abs(output))) / 2  # Normalize
        print(f"Generated sound for note: {midi_note}")
        return (midi_note, output.astype(np.int16))
