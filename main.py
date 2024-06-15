import atexit
from multiprocessing import Queue
import multiprocessing
import os
from threading import Thread
import time
import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
from tkinter import simpledialog
from tkinter import messagebox
from tensorflow import keras
import numpy as np
import sounddevice as sd
from scipy.io.wavfile import write

from scr import utils
from scr.fourier_neural_network import FourierNN
from scr.graph_canvas import GraphCanvas
from scr.simple_input_dialog import EntryWithPlaceholder, askStringAndSelectionDialog


if __name__ == "__main__":
    def main():
        def DIE(process, join_timeout=30, term_iterations=50):
            if process:
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

        process: multiprocessing.Process = None
        queue = Queue(-1)

        root = tk.Tk()
        root.rowconfigure(1, weight=1)
        root.columnconfigure(0, weight=1)
        root.columnconfigure(1, weight=1)
        root.columnconfigure(2, weight=1)

        # def draw_callback():
        #     nonlocal process
        #     print(process)
        #     if process:
        #         DIE(process)
        #         process=None
        #         train()

        graph = GraphCanvas(root, (900, 300))  # , draw_callback)
        graph.grid(row=1, column=0, columnspan=3, sticky='NSEW')

        def funny(x):
            return np.tan(np.sin(x)*np.cos(x))
        
        def funny2(x):
            return np.sin(np.cos(x) * keras.activations.elu(x))/np.cos(x)
        
        def funny3(x):
            return np.sin(np.cos(x) * keras.activations.elu(x))/np.cos(1/x)

        functions = {
            'funny': funny,
            'funny2': funny2,
            'funny3': funny3,
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
            nonlocal process
            if process:
                if process.is_alive():
                    try:
                        data = queue.get_nowait()
                        data = list(zip(graph.lst, data.reshape(-1)))
                        # graph.data=data
                        graph.draw_extern_graph_from_data(
                            data, "training", color="red")
                    except:
                        pass
                    root.after(500, train_update)
                else:
                    DIE(process)
                    process = None
                    # musik()
                    messagebox.showinfo("training Ended")

        def train():
            nonlocal fourier_nn
            nonlocal process
            if not process:
                if not fourier_nn:
                    fourier_nn = FourierNN(graph.export_data())
                else:
                    fourier_nn.update_data(graph.export_data())
                    
                # process = multiprocessing.Process(
                #     target=fourier_nn.train, args=(graph.lst, queue, True))
                process = fourier_nn.train_Process(graph.lst, queue)
                graph.draw_extern_graph_from_data(graph.export_data(), "train_data", color="blue")

                root.after(100, process.start)
                root.after(200, train_update)
                atexit.register(process.kill)
            else:
                print('already training')

        # def train():
        #     nonlocal fourier_nn
        #     nonlocal process
        #     if not process:
        #         if not fourier_nn:
        #              fourier_nn=FourierNN(graph.export_data())
        #         else:
        #             fourier_nn.update_data(graph.export_data())
        #         process=fourier_nn.train_Process(graph.lst, queue, True)
        #         print(process.pid)
        #         root.after(10, train_update)
        #         atexit.register(process.kill)
        #     else:
        #         print('already training')

        def musik():
            nonlocal fourier_nn
            midi_notes = [60, 62, 64, 65, 67, 69, 71]
            note_durations = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 1.0]
            if fourier_nn:
                notes_list=[]
                for i, (note, durration) in enumerate(zip(midi_notes,note_durations)):
                    notes_list.append(fourier_nn.synthesize_2(midi_note=note, duration=durration, sample_rate=44100))
                out=np.concatenate(notes_list)
                sd.play(out, 44100, blocking=True)

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

        # def create_new_net():
            # nonlocal fourier_nn
            # if fourier_nn:
                # fourier_nn.create_new_model()

        # def audio_load_test():
        #     nonlocal fourier_nn
        #     #filetypes = (('All files', '*.*'))
        #     filename = filedialog.askopenfilename(
        #         title='Open a file', initialdir='.', parent=root)
        #     if os.path.exists(filename):
        #         x,y=utils.process_audio(filename)
        #         data=list(zip(list(x), list(y)))
        #         graph.data=data
        #         graph._draw()
        #         if fourier_nn:
        #             fourier_nn.update_data(graph.export_data())

        button = tk.Button(root, text='Train', command=train)
        button.grid(row=2, column=0, sticky='NSEW')

        button_musik = tk.Button(root, text='Musik', command=musik)
        button_musik.grid(row=3, column=1, sticky='NSEW')

        button_clear = tk.Button(root, text='clear', command=graph.clear)
        button_clear.grid(row=2, column=2, sticky='NSEW')

        button_export = tk.Button(root, text='export', command=export)
        button_export.grid(row=3, column=0, sticky='NSEW')

        button_load = tk.Button(root, text='load', command=load)
        button_load.grid(row=2, column=1, sticky='NSEW')

        # function_input = EntryWithPlaceholder(root, "enter formular")
        # function_input.grid(row=4, column=0, columnspan=3, sticky='NSEW', padx=10)

        # def use_custom_func():
        #     func=utils.create_function(function_input.get())
        #     graph.use_preconfig_drawing(func)

        # button_new_net= tk.Button(root, text='use_custom_func', command=use_custom_func)
        # button_new_net.grid(row=3,column=2, sticky='NSEW')

        def update_2sine():
            pass

        def init():
            # utils.process_audio("C:/Users/lucas/Downloads/2-notes-octave-guitar-83275.mp3")
            graph.use_preconfig_drawing(functions['funny'])
            train()
            # train()

        root.after(500, init)

        # fourier_nn=FourierNN()

        try:
            root.mainloop()
        except KeyboardInterrupt:
            pass

        DIE(process, 2)

    main()
