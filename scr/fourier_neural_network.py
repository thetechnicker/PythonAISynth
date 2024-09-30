import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from typing import Tuple
from multiprocessing import Queue

from scr import utils
from scr.utils import QueueSTD_OUT, linear_interpolation, midi_to_freq

try:
    import torch_directml
    DIRECTML = True
except ImportError:
    DIRECTML = False

DISABLE_GPU = True


class FourierLayer(nn.Module):
    def __init__(self, fourier_degree):
        super(FourierLayer, self).__init__()
        self.tensor = torch.arange(1, fourier_degree+1)

    def forward(self, x):
        self.tensor = self.tensor.to(x.device)
        y = x.unsqueeze(1) * self.tensor
        z = torch.concat((torch.sin(y), torch.cos(y)), dim=1)
        return z


class FourierNN():
    SAMPLES = 1000
    EPOCHS = 1000
    DEFAULT_FORIER_DEGREE = 100
    FORIER_DEGREE_DIVIDER = 1
    FORIER_DEGREE_OFFSET = 0
    PATIENCE = 100
    OPTIMIZER = 'SGD'
    LOSS_FUNCTION = 'HuberLoss'
    CALC_FOURIER_DEGREE_BY_DATA_LENGTH = False

    def __init__(self, lock, data=None, stdout_queue=None):
        super(FourierNN, self).__init__()
        self.lock = lock
        self.models = []
        self.current_model = None
        self.fourier_degree = self.DEFAULT_FORIER_DEGREE
        self.orig_data_len = 0
        self.prepared_data = None
        self.stdout_queue = stdout_queue
        self.device = None
        if DIRECTML and not DISABLE_GPU:
            self.device = torch_directml.device() if torch_directml.is_available() else None
        if not self.device:
            if torch.cuda.is_available() and torch.version.hip and not DISABLE_GPU:
                self.device = torch.device('rocm')
            elif torch.cuda.is_available() and not DISABLE_GPU:
                self.device = torch.device('cuda')
            else:
                self.device = torch.device('cpu')
        print(self.device)
        if data is not None:
            self.update_data(data)

    def update_attribs(self, **kwargs):
        """
        Update model parameters.
        Accepts keyword arguments for parameters to update.
        """
        for key, val in kwargs.items():
            print(key, val)
            if hasattr(self, key):
                setattr(self, key, val)
            else:
                raise ValueError(
                    f"Parameter '{key}' does not exist in the model.")

        # If the data has been prepared, recreate the model with updated parameters
        if self.prepared_data:
            self.update_fourier_degree()
            self.create_new_model()

    def create_model(self):
        model = nn.Sequential(
            FourierLayer(self.fourier_degree),
            nn.Linear(self.fourier_degree*2, 1),  # bias=False),
            # nn.Tanh()
        )
        return model

    def update_data(self, data, stdout_queue=None):
        if stdout_queue:
            self.stdout_queue = stdout_queue
        self.prepared_data = self.prepare_data(
            list(data), std_queue=self.stdout_queue)
        if not self.current_model:
            self.current_model = self.create_model()

    def update_fourier_degree(self):
        self.fourier_degree = ((
            (self.orig_data_len // self.FORIER_DEGREE_DIVIDER) + self.FORIER_DEGREE_OFFSET)
            if self.CALC_FOURIER_DEGREE_BY_DATA_LENGTH else self.DEFAULT_FORIER_DEGREE)

    def prepare_data(self, data, std_queue: Queue = None):
        if std_queue:
            self.stdout_queue = std_queue

        self.orig_data_len = len(data)

        self.update_fourier_degree()

        x_train, y_train = linear_interpolation(
            data, self.SAMPLES, self.device)
        x_test, y_test = linear_interpolation(
            data, self.SAMPLES//2, self.device)

        return (x_train,
                y_train,
                x_test,
                y_test,
                )

    def train(self, test_data, queue=None, quiet=False, stdout_queue=None):
        # exit(-1)
        # print("WHAT THE F?")
        if stdout_queue:
            sys.stdout = QueueSTD_OUT(stdout_queue)

        print(self.OPTIMIZER,
              self.LOSS_FUNCTION,
              self.fourier_degree,
              self.PATIENCE,
              sep="\n")

        x_train_transformed, y_train, test_x, test_y = self.prepared_data

        model = self.current_model.to(self.device)
        for param in self.current_model.parameters():
            param.requires_grad = True
        model.train()

        x_train_transformed = x_train_transformed.to(self.device)
        y_train = y_train.to(self.device)
        test_x = test_x.to(self.device)
        test_y = test_y.to(self.device)

        optimizer = utils.get_optimizer(
            optimizer_name=self.OPTIMIZER,
            model_parameters=model.parameters(),
            lr=0.1)
        criterion = utils.get_loss_function(self.LOSS_FUNCTION)

        train_dataset = TensorDataset(x_train_transformed, y_train)
        train_loader = DataLoader(
            train_dataset, batch_size=int(self.SAMPLES / 2), shuffle=True)

        # prepared_test_data = torch.tensor(
        #     data=FourierNN.fourier_basis_numba(
        #         data=test_data.flatten(),
        #         indices=FourierNN.precompute_indices(self.fourier_degree)),
        #     dtype=torch.float32).to(self.device)
        prepared_test_data = torch.tensor(
            test_data.flatten(),
            dtype=torch.float32, device=self.device)
        print(prepared_test_data.shape)

        min_delta = 0.000001  # 4.337714676382401e-14
        epoch_without_change = 0
        max_loss = torch.inf

        for epoch in range(self.EPOCHS):
            # callback.on_epoch_begin(epoch)
            print(f"epoch {epoch+1} beginns")
            timestamp = time.perf_counter_ns()
            model.train()
            epoch_loss = 0
            try:
                for batch_x, batch_y in train_loader:
                    batch_x, batch_y = batch_x.to(
                        self.device), batch_y.to(self.device)
                    optimizer.zero_grad()
                    outputs = model(batch_x)
                    loss = criterion(outputs.squeeze(), batch_y)
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item()
            except Exception as e:
                print(f"Exception while training: {e}")
                exit(1)
            epoch_loss /= len(train_loader)

            # Validation
            model.eval()
            with torch.no_grad():
                val_outputs = model(test_x)
                val_loss = criterion(val_outputs.squeeze(),
                                     test_y.to(self.device))
                if queue and epoch % 10 == 0:
                    predictions = model(prepared_test_data)
                    queue.put(predictions.cpu().numpy())

            # callback.on_epoch_end(epoch, epoch_loss, val_loss.item(), model)
            time_taken = time.perf_counter_ns() - timestamp
            print(f"epoch {epoch+1} ends. "
                  f"loss: {epoch_loss:3.5f}, "
                  f"val_loss: {val_loss:3.5f}. "
                  f"Time Taken: {time_taken / 1_000_000_000}s")

            if val_loss < max_loss-min_delta:
                max_loss = epoch_loss
                epoch_without_change = 0
            else:
                epoch_without_change += 1

            if epoch_without_change == self.PATIENCE:
                print("Early Stopping")
                break

        print("Training Ended")
        self.save_model()

    def predict(self, data):
        x = torch.tensor(data, dtype=torch.float32)
        if not x.dim():
            x = x.unsqueeze(0)

        with torch.no_grad():
            y = self.current_model(x)

        return y.cpu().numpy()

    def save_model(self, filename='./tmp/model.pth'):
        torch.save(self.current_model.state_dict(), filename)

    def load_new_model_from_file(self, filename='./tmp/model.pth', *, delete_tmp=False):
        self.current_model = self.create_model()
        model = torch.load(filename, weights_only=False)
        if delete_tmp:
            os.remove(filename)
        print(model)
        self.current_model.load_state_dict(model)
        self.current_model.eval()
        print(self.current_model)

    def create_new_model(self):
        if hasattr(self, "prepared_data"):
            if self.prepared_data:
                self.current_model = self.create_model()

    def change_model(self, net_no):
        index = net_no - 1
        if index >= 0:
            new_model = self.models.pop(index)
            if self.current_model:
                self.models.insert(index, self.current_model)
            self.current_model = new_model
