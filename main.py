from _version import version
from scr.utils import DIE
from scr.simple_input_dialog import askStringAndSelectionDialog
from scr.graph_canvas import GraphCanvas
from scr.fourier_neural_network_gui import NeuralNetworkGUI
from scr.fourier_neural_network import FourierNN
from scr import music
import matplotlib.pyplot as plt
import atexit
import gc
import multiprocessing
from multiprocessing import Queue
import os
from threading import Thread

from scr import utils
import random
import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
from tkinter import messagebox
from tensorflow import keras
import numpy as np
import matplotlib

# from tmp.std_redirect import RedirectedOutputFrame
matplotlib.use('TkAgg')


def main():
    trainings_process: multiprocessing.Process = None
    training_started = False
    queue = Queue(-1)

    root = tk.Tk()
    root.title = f"AI Synth - {version}"
    root.rowconfigure(1, weight=1)
    root.columnconfigure(0, weight=1)
    root.columnconfigure(1, weight=1)
    root.columnconfigure(2, weight=1)
    root.columnconfigure(3, weight=1)


    # def draw_callback():
    #     nonlocal process
    #     print(process)
    #     if process:
    #         DIE(process)
    #         process=None
    #         train()

    def stupid_but_ram():
        nonlocal root
        gc.collect()
        root.after(5*1000, stupid_but_ram)

    graph = GraphCanvas(root, (900, 300))  # , draw_callback)
    graph.grid(row=1, column=0, columnspan=3, sticky='NSEW')

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

    functions = {
        'funny': funny,
        'funny2': funny2,
        'funny3': funny3,
        'random': my_random,
        'cool': my_complex_function,
        'bing': my_generated_function,
        'nice': nice,
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

    n = tk.StringVar()

    def predefined_functions(event):
        graph.use_preconfig_drawing(functions[n.get()])

    label = tk.Label(root, text='Predifined functions')
    label.grid(row=0, column=0, sticky='NSEW')

    predefined_functions_select = ttk.Combobox(
        root, textvariable=n, state="readonly")
    predefined_functions_select.bind(
        '<<ComboboxSelected>>', predefined_functions)
    predefined_functions_select.grid(
        row=0, column=1, columnspan=2, sticky='NSEW')
    predefined_functions_select['values'] = list(functions.keys())

    fourier_nn = None

    def train_update():
        nonlocal trainings_process
        if trainings_process:
            if trainings_process.is_alive():
                try:
                    data = queue.get_nowait()
                    data = list(zip(graph.lst, data.reshape(-1)))
                    # graph.data=data
                    graph.draw_extern_graph_from_data(
                        data, "training", color="red", width=graph.point_radius/4)
                except:
                    pass
                root.after(100, train_update)
            else:
                exit_code = trainings_process.exitcode
                if exit_code == 0:
                    print("loading trained model")
                    fourier_nn.load_tmp_model()
                    print("model loaded")
                    graph.draw_extern_graph_from_func(
                        fourier_nn.predict, "training", color="red", width=graph.point_radius/4)  # , graph_type='crazy'
                DIE(trainings_process)
                trainings_process = None
                training_started = False
                messagebox.showinfo(
                    "training Ended", f"exit code: {exit_code}")

    def init_or_update_nn():
        nonlocal fourier_nn
        if not fourier_nn:
            fourier_nn = FourierNN(graph.export_data())
        else:
            fourier_nn.update_data(graph.export_data())
        fourier_nn.save_tmp_model()
        start_training()

    def start_training():
        nonlocal trainings_process
        trainings_process = multiprocessing.Process(
            target=fourier_nn.train, args=(graph.lst, queue,))
        root.after(100, trainings_process.start)
        root.after(200, train_update)

    def train():
        nonlocal trainings_process
        nonlocal training_started
        if trainings_process or training_started:
            print('already training')
            return
        training_started = True
        print("STARTING TRAINING")
        t = Thread(target=init_or_update_nn)
        t.daemon = True
        t.start()
        atexit.register(DIE, trainings_process)
        graph.draw_extern_graph_from_data(
            graph.export_data(), "train_data", color="blue")

    def export():
        default_format = 'keras'
        nonlocal fourier_nn
        if fourier_nn:
            path = './tmp'
            if not os.path.exists(path):
                os.mkdir(path)
            i = 0
            while os.path.exists(path+f"/model{i}.{default_format}"):
                i += 1

            dialog = askStringAndSelectionDialog(parent=root,
                                                 title="Save Model",
                                                 label_str="Enter a File Name",
                                                 default_str=f"model{i}",
                                                 label_select="Select Format",
                                                 default_select=default_format,
                                                 values_to_select_from=["keras", "h5"])
            name, file_format = dialog.result
            if not name:
                name = f"model{i}"

            file = f"{path}/{name}.{file_format}"
            print(file)
            if not os.path.exists(file):
                try:
                    fourier_nn.save_model(file)
                except Exception as e:
                    messagebox.showwarning(
                        "ERROR - Can't Save Model", f"{e}")
            else:
                messagebox.showwarning(
                    "File already Exists", f"The selected filename {name} already exists.")

    def load():
        nonlocal fourier_nn
        filetypes = (('Keras files', '*.keras'),
                     ('HDF5 files', '*.h5'), ('All files', '*.*'))
        filename = filedialog.askopenfilename(
            title='Open a file', initialdir='.', filetypes=filetypes, parent=root)
        if os.path.exists(filename):
            if not fourier_nn:
                fourier_nn = FourierNN(data=None)
            fourier_nn.load_new_model_from_file(filename)
            name = os.path.basename(filename).split('.')[0]
            graph.draw_extern_graph_from_func(fourier_nn.predict, name)
            print(name)
            fourier_nn.update_data(data=graph.get_graph(name=name)[0])

    button = tk.Button(root, text='start training', command=train)
    button.grid(row=2, column=0, sticky='NSEW')

    def command(): return music.midi_to_musik_live(root, fourier_nn)
    button_musik = tk.Button(
        root, text='play music from midi', command=command)
    button_musik.grid(row=3, column=1, sticky='NSEW')

    button_clear = tk.Button(root, text='clear graph', command=graph.clear)
    button_clear.grid(row=2, column=2, sticky='NSEW')

    button_export = tk.Button(root, text='export neural net', command=export)
    button_export.grid(row=3, column=0, sticky='NSEW')

    button_load = tk.Button(root, text='load neural net', command=load)
    button_load.grid(row=2, column=1, sticky='NSEW')

    def clear():
        nonlocal fourier_nn
        if fourier_nn:
            graph.draw_extern_graph_from_func(
                fourier_nn.predict, "training", color="red", width=graph.point_radius/4)
    # button can be used for other things
    button_new_net = tk.Button(
        root, text='redraw graph from neural net', command=clear)
    button_new_net.grid(row=3, column=2, sticky='NSEW')

    def init():
        graph.use_preconfig_drawing(functions['random'])
        train()

    defaults = {
        'SAMPLES': 44100//2,
        'EPOCHS': 100,
        'DEFAULT_FORIER_DEGREE': 250,
        'FORIER_DEGREE_DIVIDER': 1,
        'FORIER_DEGREE_OFFSET': 0,
        'PATIENCE': 10,
    }

    def update_frourier_params(key, value):
        nonlocal fourier_nn
        if not fourier_nn:
            fourier_nn = FourierNN()
        fourier_nn.update_attribs(**{key: value})
    gui = NeuralNetworkGUI(root, defaults=defaults,
                           callback=update_frourier_params)
    gui.grid(row=0, column=3, rowspan=3, sticky='NSEW')

    # root.after(500, init)

    try:
        root.mainloop()
    except KeyboardInterrupt:
        pass

    if trainings_process:
        DIE(trainings_process, 2)


if __name__ == "__main__":
    os.environ['HAS_RUN_INIT'] = 'True'
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    multiprocessing.set_start_method("spawn")

    main()
