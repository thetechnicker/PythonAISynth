import numpy as np


def midi_to_freq(midi_note):
    return 440.0 * np.power(2.0, (midi_note - 69) / 12.0)


def pair_iterator(lst):
    for i in range(len(lst) - 1):
        yield lst[i], lst[i+1]

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
