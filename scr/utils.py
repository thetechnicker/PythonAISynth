import atexit
from io import StringIO
from multiprocessing import Process, Queue
import random
import sys
import threading
import time
from typing import Any, Callable, TextIO
import numpy as np
from scipy.fft import dst


class QueueSTD_OUT(TextIO):
    def __init__(self, queue: Queue) -> None:
        super().__init__()
        self.queue = queue

    def write(self, msg):
        self.queue.put(msg)


def DIE(process: Process, join_timeout=30, term_iterations=50):
    if process:
        if not process.is_alive():
            return
        print("Attempting to stop process (die)", flush=True)
        process.join(join_timeout)
        i = 0
        while process.is_alive() and i < term_iterations:
            print("Sending terminate signal (DIE)", flush=True)
            process.terminate()
            time.sleep(0.5)
            i += 1
        while process.is_alive():
            print("Sending kill signal !!!(DIE!!!)", flush=True)
            process.kill()
            time.sleep(0.1)
        print("Success")


def messure_time_taken(name, func, *args, **kwargs):
    timestamp = time.perf_counter_ns()
    result = func(*args, **kwargs)
    print(
        f"Time taken for {name}: {(time.perf_counter_ns()-timestamp)/1_000_000_000}s")
    input("paused")
    return result


def run_multi_arg_func(func, *args, **kwargs):
    func(*args, **kwargs)


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


def interpolate(point1, point2, t=0.5):
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
        get_prepared_random_color.colors = []
        for i in range(maxColors or 100):
            while True:
                color = random_hex_color()
                if not color in get_prepared_random_color.colors:
                    get_prepared_random_color.colors.append(color)
                    break
    random.shuffle(get_prepared_random_color.colors)
    if len(get_prepared_random_color.colors) > 0:
        return get_prepared_random_color.colors.pop()
    else:
        raise Exception("No more unique values left to generate")


def lighten_color(r, g, b, percent):
    """Lighten a color by a certain percentage."""
    r_lighter = min(255, int(r + (255 - r) * percent))
    g_lighter = min(255, int(g + (255 - g) * percent))
    b_lighter = min(255, int(b + (255 - b) * percent))
    return r_lighter, g_lighter, b_lighter


def exec_with_queue_stdout(func: Callable, *args, queue: Queue):
    sys.stdout = QueueSTD_OUT(queue=queue)
    return func(*args)


def calculate_peak_frequency(signal):
    np.savetxt('tmp/why_do_i_need_this_array.txt', signal)
    test_data = np.loadtxt('tmp/why_do_i_need_this_array.txt')
    test_data = test_data - np.mean(test_data)

    # Perform the Discrete Sine Transform (DST)
    transformed_signal = dst(test_data, type=4, norm='ortho')

    peak_index = np.argmax(np.abs(transformed_signal))
    peak_freq = (peak_index+1)/2
    return peak_freq


def run_after_ms(delay_ms: int, func: Callable[..., Any], *args: Any, **kwargs: Any) -> None:
    """
    Schedules a function to be run after a specified delay in milliseconds.

    Args:
        delay_ms (int): The delay in milliseconds before the function is executed.
        func (Callable[..., Any]): The function to be executed after the delay.
        *args (Any): Positional arguments to pass to the function.
        **kwargs (Any): Keyword arguments to pass to the function.
    """
    delay_seconds = delay_ms / 1000.0
    timer = threading.Timer(delay_seconds, func, args=args, kwargs=kwargs)
    timer.start()


def kill_timer(timer):
    if timer.is_alive():
        timer.cancel()
