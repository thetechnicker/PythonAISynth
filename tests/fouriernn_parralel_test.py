import multiprocessing
import numpy as np
# import tensorflow as tf
# import matplotlib
# matplotlib.use('TkAgg')
from context import scr
from scr.fourier_neural_network import FourierNN
from scr.utils import DIE

# TODO redo
# if __name__ == '__main__':
#     multiprocessing.set_start_method("spawn")
#     # Test the class with custom data points
#     data = [(x, np.sin(x * keras.activations.relu(x)))
#             for x in np.linspace(np.pi, -np.pi, 100)]
#     fourier_nn = FourierNN(data)

#     proc=fourier_nn.train_Process(np.linspace(np.pi, -np.pi, 100))
#     proc.start()
#     while proc.is_alive():
#         pass
#     DIE(proc)
