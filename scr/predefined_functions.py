import random
from tensorflow import keras
import numpy as np

from scr import utils


def funny(x):
    return np.tan(np.sin(x)*np.cos(x))


def funny2(x):
    return np.sin(np.cos(x) * keras.activations.elu(x))/np.cos(x)


def funny3(x):
    return np.sin(np.cos(x) * keras.activations.elu(x))/np.cos(1/x)


def my_random(x):
    return np.sin(x*random.uniform(-1, 1))


def my_complex_function(x):
    if x > 0:
        return np.sin(x) * (np.sin(np.tan(x) * keras.activations.relu(x)) / np.cos(random.uniform(-1, 1) * x))
    else:
        x = -x
        return -np.sin(x) * (np.sin(np.tan(x) * keras.activations.relu(x)) / np.cos(random.uniform(-1, 1) * x))


def nice(x):
    x_greater_pi = False
    x = utils.map_value(x, -np.pi, np.pi, 0, 2*np.pi)
    if x >= 2*np.pi:
        x_greater_pi = True
        x = x-np.pi
    if x > (np.pi/2):
        y = np.sin(np.tan(x))
    elif 0 < x and x < (np.pi/2):
        y = np.cos(-np.tan(x))
    elif (-np.pi/2) < x and x < 0:
        y = np.cos(np.tan(x))
    else:
        y = np.sin(-np.tan(x))
    if x_greater_pi:
        return -y
    return y


def my_generated_function(x):
    if x > np.pi:
        return np.sin(np.tan(x) * keras.activations.softplus(x)) / np.sin(random.uniform(-1, 1) * x)
    elif 0 < x and x <= np.pi:
        return np.cos(np.sin(x) * keras.activations.softplus(-x)) / np.cos(random.uniform(-1, 1) * x)
    elif -np.pi < x and x <= 0:
        return np.cos(np.sin(x) * keras.activations.softplus(x)) / np.sin(random.uniform(-1, 1) * x)
    else:
        return np.sin(np.tan(-x) * keras.activations.softplus(x)) / np.cos(random.uniform(-1, 1) * x)


def extreme(x):
    if not hasattr(extreme, 'prev'):
        extreme.prev = -1
    else:
        extreme.prev = -extreme.prev
    return extreme.prev


predefined_functions_dict = {
    'funny': funny,
    'funny2': funny2,
    'funny3': funny3,
    'random': my_random,
    'cool': my_complex_function,
    'bing': my_generated_function,
    'nice': nice,
    'extreme': extreme,
    'sin': np.sin,
    'cos': np.cos,
    'tan': np.tan,
    'relu': keras.activations.relu,
    'elu': keras.activations.elu,
    'linear': keras.activations.linear,
    'sigmoid': keras.activations.sigmoid,
    'exponential': keras.activations.exponential,
    'selu': keras.activations.selu,
    'gelu': keras.activations.gelu,
}
