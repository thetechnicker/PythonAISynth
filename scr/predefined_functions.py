import random
import numpy as np

from scr import utils


def funny(x):
    return np.tan(np.sin(x) * np.cos(x))


def funny2(x):
    return np.sin(np.cos(x) * np.maximum(0, x)) / np.cos(x)


def funny3(x):
    return np.sin(np.cos(x) * np.maximum(0, x)) / np.cos(1/x)


def my_random(x):
    x = x-np.pi
    return np.sin(x * np.random.uniform(-1, 1, size=x.shape))


def my_complex_function(x):
    x = np.abs(x)  # Ensure x is non-negative for relu
    return np.where(x > 0,
                    np.sin(x) * (np.sin(np.tan(x) * x) /
                                 np.cos(np.random.uniform(-1, 1, size=x.shape) * x)),
                    -np.sin(-x) * (np.sin(np.tan(-x) * -x) / np.cos(np.random.uniform(-1, 1, size=x.shape) * -x)))


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
    part1 = np.sin(x) * np.maximum(0, x)
    part2 = np.cos(x) * (1 / (1 + np.exp(-x)))  # Sigmoid approximation
    part3 = np.tan(x) * np.tanh(x)
    part4 = np.exp(x) * np.log1p(np.exp(x))  # Softplus approximation

    # Combine the parts to create the final output
    # Adding a small value to avoid division by zero
    result = part1 + part2 - part3 / (part4 + 1e-7)
    return result


def extreme(x):
    y = np.tile(np.array([-1, 1]), len(x)//2).flatten()
    if len(y) < len(x):
        y = np.append(y, [y[-2]])
    return y


predefined_functions_dict = {
    'funny': funny,
    'funny2': funny2,
    'funny3': funny3,
    'random': my_random,
    'cool': my_complex_function,
    'bing': my_generated_function,
    # 'nice': nice,
    'extreme': extreme,
    'sin': np.sin,
    'cos': np.cos,
    'tan': np.tan,
    'relu': lambda x: np.maximum(0, x - np.pi),
    # ELU approximation
    'elu': lambda x: np.where(x - np.pi > 0, x - np.pi, np.expm1(x - np.pi)),
    'linear': lambda x: x - np.pi,  # Linear function
    'sigmoid': lambda x: 1 / (1 + np.exp(-(x - np.pi))),
    'exponential': lambda x: np.exp(x - np.pi),
    # SELU approximation
    'selu': lambda x: np.where(x - np.pi > 0, 1.0507 * (x - np.pi), 1.0507 * (np.exp(x - np.pi) - 1)),
    # GELU approximation
    'gelu': lambda x: 0.5 * (x - np.pi) * (1 + np.tanh(np.sqrt(2 / np.pi) * ((x - np.pi) + 0.044715 * np.power((x - np.pi), 3))))
}


def call_func(name, x):
    return np.clip(predefined_functions_dict[name](x), -1, 1)
