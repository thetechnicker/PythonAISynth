import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.callbacks import Callback

from scr.fourier_neural_network import FourierNN

def train_async_test(test_data, queue, *, fourier_nn:FourierNN=None, train_data=None):
    if not fourier_nn:
        if not train_data:
            raise ValueError("if fourier_nn is not defined you must set train_data")
        fourier_nn=FourierNN(train_data)
    fourier_nn.create_new_model()
    fourier_nn.train(test_data, queue)
