import atexit
from io import StringIO
from multiprocessing import Process, Queue
import random
import sys
import threading
import time
from typing import Any, Callable, TextIO
import numpy as np
import psutil
from scipy.fft import dst
from numba import njit
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import tkinter as tk


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


def calculate_max_batch_size(num_features, dtype=np.float32, memory_buffer=0.1):
    # Determine the number of bytes per feature based on the data type
    bytes_per_feature = np.dtype(dtype).itemsize

    # Calculate memory required for one sample
    memory_per_sample = num_features * bytes_per_feature

    # Get available memory in bytes
    available_memory = psutil.virtual_memory().available

    # Calculate maximum batch size with a memory buffer
    max_batch_size = (memory_buffer * available_memory) // memory_per_sample

    return max_batch_size


def messure_time_taken(name, func, *args, wait=True, **kwargs):
    if not hasattr(messure_time_taken, 'time_taken'):
        messure_time_taken.time_taken = {}
    if name not in messure_time_taken.time_taken:
        messure_time_taken.time_taken[name] = 0
    timestamp = time.perf_counter_ns()
    result = func(*args, **kwargs)
    time_taken = (time.perf_counter_ns()-timestamp)/1_000_000_000
    print(
        f"Time taken for {name}: {time_taken:6.6f}s")
    if wait:
        input("paused")
    messure_time_taken.time_taken[name] = max(
        messure_time_taken.time_taken[name], time_taken)
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


@ njit
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


@ njit
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


timers = []


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
    timer.setDaemon(True)
    timer.start()
    timers.append(timer)  # Keep track of the timer


def kill_timer(timer: threading.Timer):
    if timer.is_alive():
        timer.cancel()


def cleanup_timers():
    for timer in timers:
        kill_timer(timer)


atexit.register(cleanup_timers)


def tk_after_errorless(master: tk.Tk, delay_ms: int, func: Callable[..., Any], *args: Any):
    def safe_call(master: tk.Tk, func: Callable[..., Any], *args: Any):
        if master.winfo_exists():
            func(*args)

    try:
        if master.winfo_exists():
            master.update_idletasks()
            master.after(delay_ms, safe_call, master, func, *args)
    except:
        pass


def get_optimizer(optimizer_name, model_parameters, **kwargs):
    if hasattr(optim, optimizer_name):
        optimizer_class = getattr(optim, optimizer_name)
        if issubclass(optimizer_class, optim.Optimizer):
            return optimizer_class(model_parameters, **kwargs)
    raise ValueError(f"Optimizer '{optimizer_name}' not found in torch.optim")


def get_loss_function(loss_name, **kwargs):
    if hasattr(nn, loss_name):
        loss_class = getattr(nn, loss_name)
        if issubclass(loss_class, nn.modules.loss._Loss):
            return loss_class(**kwargs)
    raise ValueError(f"Loss function '{loss_name}' not found in torch.nn")


def linear_interpolation(data, target_length, device='cpu'):
    # Convert data to a tensor and move to the specified device
    data = np.array(data)
    data_tensor = torch.tensor(data, dtype=torch.float32).to(device)

    # Extract x and y values
    x_values = data_tensor[:, 0]
    y_values = data_tensor[:, 1]

    # Generate new x values
    new_x_values = torch.linspace(
        x_values.min(), x_values.max(), target_length).to(device)

    # Perform linear interpolation
    new_y_values = F.interpolate(y_values.unsqueeze(0).unsqueeze(0),
                                 size=new_x_values.size(0),
                                 mode='linear',
                                 align_corners=True).squeeze()

    return (new_x_values.cpu(), new_y_values.cpu(),)


def center_and_scale(arr):
    # Convert the input list to a NumPy array
    arr = np.array(arr)

    # Center the array around 0
    centered_arr = arr - np.mean(arr)

    # Clip the centered array to the range [-1, 1]
    clipped_arr = np.clip(centered_arr, -1, 1)

    # Scale the values to fit exactly within [-1, 1]
    min_val = np.min(centered_arr)
    max_val = np.max(centered_arr)

    # Avoid division by zero if all values are the same
    if max_val - min_val == 0:
        # or return clipped_arr if you prefer
        return np.zeros_like(centered_arr)

    # Scale to [-1, 1]
    scaled_arr = 2 * (centered_arr - min_val) / (max_val - min_val) - 1
    return scaled_arr


def wrap_concat(tensor, idx1, idx2):
    length = tensor.size(0)

    if idx2 >= length:
        idx2 = idx2 % length

    if idx1 <= idx2:
        result = tensor[idx1:idx2]
    else:
        result = torch.cat((tensor[idx1:], tensor[:idx2]))
    # print(result.shape)
    return result
