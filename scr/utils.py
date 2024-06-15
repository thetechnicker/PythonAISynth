import random
import re
import time
from matplotlib import pyplot as plt
import numpy as np
import sympy
from sympy.utilities.lambdify import lambdify
from tensorflow.keras import activations
# import librosa
# from scipy.fft import *
# from scipy.io import wavfile


def messure_time_taken(name, func, *args, **kwargs):
    timestamp=time.perf_counter_ns()
    func(*args, **kwargs)
    print(f"Time taken for {name}: {(time.perf_counter_ns()-timestamp)/1_000_000_000}s")

def midi_to_freq(midi_note):
    return 440.0 * np.power(2.0, (midi_note - 69) / 12.0)


def pair_iterator(lst):
    for i in range(len(lst) - 1):
        yield lst[i], lst[i+1]

def note_iterator(lst):
    for i in range(len(lst)-1):
        yield lst[i], lst[i+1]
    yield lst[len(lst)-1], None

def find_two_closest(num_list, x):
    # Sort the list in ascending order based on the absolute difference with x
    sorted_list = sorted(num_list, key=lambda num: abs(num - x))
    
    # Return the first two elements from the sorted list
    return sorted_list[:2]

def interpolate(point1, point2, t):
    """
    Interpolate between two 2D points.

    Parameters:
    point1 (tuple): The first point as a tuple (x1, y1).
    point2 (tuple): The second point as a tuple (x2, y2).
    t (float): The interpolation parameter. When t=0, returns point1. When t=1, returns point2.

    Returns:
    tuple: The interpolated point as a tuple (x, y).
    """
    x1, y1 = point1
    x2, y2 = point2

    x = x1 * (1 - t) + x2 * t
    y = y1 * (1 - t) + y2 * t

    return (x, y)

def interpolate_vectorized(point1, point2, t_values):
    """
    Vectorized interpolation between two 2D points.

    Parameters:
    point1 (tuple): The first point as a tuple (x1, y1).
    point2 (tuple): The second point as a tuple (x2, y2).
    t_values (numpy.ndarray): An array of interpolation parameters. For each t in t_values, when t=0, returns point1. When t=1, returns point2.

    Returns:
    numpy.ndarray: An array of interpolated points. Each point is a tuple (x, y).
    """
    point1 = np.array(point1)
    point2 = np.array(point2)

    points = (1 - t_values[:, None]) * point1 + t_values[:, None] * point2

    return points

def is_in_interval(value, a, b):
    lower_bound = min(a, b)
    upper_bound = max(a, b)
    return lower_bound <= value <= upper_bound

def map_value(value, leftMin, leftMax, rightMin, rightMax):
    leftSpan = leftMax - leftMin
    rightSpan = rightMax - rightMin
    valueScaled = float(value - leftMin) / float(leftSpan)
    return rightMin + (valueScaled * rightSpan)

def map_value_old(x, a1, a2, b1, b2):
    """
    Maps a value from one range to another.

    Parameters:
    x (float): The value to map.
    a1 (float): The lower bound of the original range.
    a2 (float): The upper bound of the original range.
    b1 (float): The lower bound of the target range.
    b2 (float): The upper bound of the target range.

    Returns:
    float: The mapped value in the target range.
    """
    return b1 + ((x - a1) * (b2 - b1)) / (a2 - a1)


def random_hex_color():
    # Generate a random color in hexadecimal format
    color = '{:06x}'.format(random.randint(0x111111, 0xFFFFFF))
    return '#' + color

def random_color():
    colors = ["lime", "orange", "yellow", "green", "blue", "indigo", "violet"]
    return random.choice(colors)

def get_prepared_random_color(maxColors=None):
    if not hasattr(get_prepared_random_color, 'colors'):
        get_prepared_random_color.colors=[]
        for i in range(maxColors or 100):
            while True:
                color=random_hex_color()
                if not color in get_prepared_random_color.colors:
                    get_prepared_random_color.colors.append(color)
                    break
    random.shuffle(get_prepared_random_color.colors)
    if len(get_prepared_random_color.colors)>0:
        return get_prepared_random_color.colors.pop()
    else:
        raise Exception("No more unique values left to generate")
    
# # (sin(x * elu(x)) * relu(x)) / 1 / tan(1 / cos(1 / gelu(x)))
# def create_function(formula):
#     # Define the symbols
#     x = sympy.symbols('x')

#     # Define a dictionary mapping activation function names to the actual functions
#     activation_funcs = [name for name in dir(activations) if callable(getattr(activations, name))]

#     # Replace activation function names in the formula with the actual functions
#     for name in activation_funcs:
#         # formula = formula.replace(name, f'activation_funcs["{name}"]')
#         formula=re.sub(r'\b' + name + r'\b', f'activations.{name}', formula)
    
#     print(formula)

#     # Convert the formula to a sympy expression
#     expr = sympy.sympify(formula)

#     # Convert the sympy expression to a Python function
#     f = lambdify(x, expr, "numpy")

#     # Return the created function
#     return f

# def get_freq(sr, data, start_time, end_time):

#     # Open the file and convert to mono
#     if data.ndim > 1:
#         data = data[:, 0]
#     else:
#         pass

#     # Return a slice of the data from start_time to end_time
#     dataToRead = data[int(start_time * sr / 1000) : int(end_time * sr / 1000) + 1]

#     # Fourier Transform
#     N = len(dataToRead)
#     yf = rfft(dataToRead)
#     xf = rfftfreq(N, 1 / sr)

#     # Uncomment these to see the frequency spectrum as a plot
#     # plt.plot(xf, np.abs(yf))
#     # plt.show()

#     # Get the most dominant frequency and return it
#     idx = np.argmax(np.abs(yf))
#     freq = xf[idx]
#     return freq

# def process_audio(file_path):
    # data, sr = librosa.load(file_path)
    # start=100

    # freq=get_freq(sr, data, start, sr*100)

    # print(freq)

    # end=int(sr/freq)
    # print(end)
    # y=data[start:start+end]
    # # Load the audio file
    # x1 = np.linspace(-np.pi, np.pi, len(data))
    # x2 = np.linspace(-np.pi, np.pi, len(y))
    
    # plt.plot(x1, data, label="asdf")
    # plt.plot(x2,y)
    # plt.legend()
    # plt.show()

    # # Map x to a range from -pi to pi

    # # Map y to a range of -1 to 1
    # # y = 2 * (y - np.min(y)) / np.ptp(y) - 1

    # return x2, y