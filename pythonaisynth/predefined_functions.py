import random
import numpy as np

from pythonaisynth import utils


def tan_sin_cos(x):
    return np.tan(np.sin(x) * np.cos(x))


def sin_cos_relu_div_cos(x):
    return np.sin(np.cos(x) * np.maximum(0, x)) / np.cos(x)


def sin_cos_relu_div_cos_offset(x):
    return np.sin(np.cos(x) * np.maximum(0, x)) / np.cos(1 / (x + 0.001))


def random_sin(x):
    x = x - np.pi
    return np.sin(x * np.random.uniform(-1, 1, size=x.shape))


def complex_trig_random(x):
    x = np.abs(x)  # Ensure x is non-negative for relu
    return np.where(
        x > 0,
        np.sin(x)
        * (np.sin(np.tan(x) * x) / np.cos(np.random.uniform(-1, 1, size=x.shape) * x)),
        -np.sin(-x)
        * (
            np.sin(np.tan(-x) * -x)
            / np.cos(np.random.uniform(-1, 1, size=x.shape) * -x)
        ),
    )


def conditional_trig(x):
    x_greater_pi = False
    if x >= 2 * np.pi:
        x_greater_pi = True
        x = x - np.pi
    if x > (np.pi / 2):
        y = np.sin(np.tan(x))
    elif 0 < x and x < (np.pi / 2):
        y = np.cos(-np.tan(x))
    elif (-np.pi / 2) < x and x < 0:
        y = np.cos(np.tan(x))
    else:
        y = np.sin(-np.tan(x))
    if x_greater_pi:
        return -y
    return y


def vectorized_conditional_trig(x):
    x = np.array(x)
    x_greater_pi = x >= 2 * np.pi
    x = np.where(x_greater_pi, x - np.pi, x)

    y = np.where(
        x > (np.pi / 2),
        np.sin(np.tan(x)),
        np.where(
            (0 < x) & (x < (np.pi / 2)),
            np.cos(-np.tan(x)),
            np.where(
                ((-np.pi / 2) < x) & (x < 0), np.cos(np.tan(x)), np.sin(-np.tan(x))
            ),
        ),
    )

    y = np.where(x_greater_pi, -y, y)
    return y


def combined_trig_activation(x):
    # Apply a combination of trigonometric functions and activation functions
    part1 = np.sin(x) * np.maximum(0, x)
    part2 = np.cos(x) * (1 / (1 + np.exp(-x)))  # Sigmoid approximation
    part3 = np.tan(x) * np.tanh(x)
    part4 = np.exp(x) * np.log1p(np.exp(x))  # Softplus approximation

    # Combine the parts to create the final output
    # Adding a small value to avoid division by zero
    result = part1 + part2 - part3 / (part4 + 1e-7)
    return result


def alternating_sign_pattern(x):
    y = np.tile(np.array([-1, 1]), len(x) // 2).flatten()
    if len(y) < len(x):
        y = np.append(y, [y[-2]])
    return y


predefined_functions_dict = {
    "Func1 (sin_cos_relu_div_cos)": sin_cos_relu_div_cos,
    "Func2 (sin_cos_relu_div_cos)": random_sin,
    "Func3 (vectorized_conditional_trig)": vectorized_conditional_trig,

    "sin": np.sin,
    "cos": np.cos,
    "relu": lambda x: np.maximum(0, x - np.pi),
    # ELU approximation
    "elu": lambda x: np.where(x - np.pi > 0, x - np.pi, np.expm1(x - np.pi)),
    "linear": lambda x: x - np.pi,  # Linear function
    "sigmoid": lambda x: 1 / (1 + np.exp(-(x - np.pi))),
    "exponential": lambda x: np.exp(x - np.pi),
    # SELU approximation
    "selu": lambda x: np.where(
        x - np.pi > 0, 1.0507 * (x - np.pi), 1.0507 * (np.exp(x - np.pi) - 1)
    ),
    # GELU approximation
    "gelu": lambda x: 0.5
    * (x - np.pi)
    * (
        1
        + np.tanh(
            np.sqrt(2 / np.pi) * ((x - np.pi) +
                                  0.044715 * np.power((x - np.pi), 3))
        )
    ),
}


def call_func(name, x):
    return np.clip(predefined_functions_dict[name](x), -1, 1)
