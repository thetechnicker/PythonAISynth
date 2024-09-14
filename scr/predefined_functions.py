import random
import numpy as np
import torch

from scr import utils


def funny(x):
    return np.tan(np.sin(x) * np.cos(x))


def funny2(x):
    return np.sin(np.cos(x) * torch.nn.functional.elu(torch.tensor(x)).numpy()) / np.cos(x)


def funny3(x):
    return np.sin(np.cos(x) * torch.nn.functional.elu(torch.tensor(x)).numpy()) / np.cos(1/x)


def my_random(x):
    return np.sin(x * random.uniform(-1, 1))


def my_complex_function(x):
    if x > 0:
        return np.sin(x) * (np.sin(np.tan(x) * torch.nn.functional.relu(torch.tensor(x)).numpy()) / np.cos(random.uniform(-1, 1) * x))
    else:
        x = -x
        return -np.sin(x) * (np.sin(np.tan(x) * torch.nn.functional.relu(torch.tensor(x)).numpy()) / np.cos(random.uniform(-1, 1) * x))


def nice(x):
    x_greater_pi = False
    x = utils.map_value(x, -np.pi, np.pi, 0, 2*np.pi)
    if x >= 2*np.pi:
        x_greater_pi = True
        x = x - np.pi
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
    # Apply a combination of trigonometric functions and activation functions
    part1 = np.sin(x) * torch.nn.functional.relu(torch.tensor(x)).numpy()
    part2 = np.cos(x) * torch.sigmoid(torch.tensor(x)).numpy()
    part3 = np.tan(x) * torch.tanh(torch.tensor(x)).numpy()
    part4 = np.exp(x) * torch.nn.functional.softplus(torch.tensor(x)).numpy()

    # Combine the parts to create the final output
    # Adding a small value to avoid division by zero
    result = part1 + part2 - part3 / (part4 + 1e-7)
    return result


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
    'relu': torch.nn.functional.relu,
    'elu': torch.nn.functional.elu,
    'linear': torch.nn.functional.linear,
    'sigmoid': torch.sigmoid,
    'exponential': torch.exp,
    'selu': torch.nn.functional.selu,
    'gelu': torch.nn.functional.gelu,
}
