import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.callbacks import Callback

from scr.fourier_neural_network import FourierNN

def train_async_test(fourier_n:FourierNN, test_data, queue):
    fourier_n.create_new_model()
    fourier_n.train(test_data, queue)