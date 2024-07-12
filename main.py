import atexit
import gc
import multiprocessing
if __name__=='__main__':
    multiprocessing.set_start_method("spawn")
from multiprocessing import Queue
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import sys
from threading import Thread

from scr import utils
import random
import time
import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
from tkinter import messagebox
from tensorflow import keras
import numpy as np
import sounddevice as sd

# class AsyncBuffer:
#     def __init__(self):
#         self.queue = Queue()
#         self.thread = Thread(target=self._monitor_queue, daemon=True)
#         self.thread.start()

#     def _monitor_queue(self):
#         while True:
#             item:str = self.queue.get()
#             item=item.replace("\n", "")
#             if item is None:
#                 break
#             if item=='':
#                 continue
#             print(item, file=sys.__stdout__, end="\n")  # print to the real stdout

#     def write(self, data):
#         self.queue.put(data)
    
#     def flush(self):
#         pass

#     def exit(self):
#         sys.stdout=sys.__stdout__
#         self.queue.put(None)
#         self.thread.join()

# sys.stdout=AsyncBuffer()


from scr import music
from scr.fourier_neural_network import FourierNN
from scr.graph_canvas import GraphCanvas
from scr.simple_input_dialog import askStringAndSelectionDialog
from scr.utils import DIE

if __name__ == "__main__":
    def main():

        trainings_process: multiprocessing.Process = None
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
            return np.sin(x*random.uniform(-1,1))
        
        def my_complex_function(x):
            if x>0:
                return np.sin(x) * (np.sin(np.tan(x) * keras.activations.relu(x)) / np.cos(random.uniform(-1, 1) * x))
            else:
                x=-x
                return -np.sin(x) * (np.sin(np.tan(x) * keras.activations.relu(x)) / np.cos(random.uniform(-1, 1) * x))
            
        def nice(x):
            x_greater_pi=False
            x=utils.map_value(x, -np.pi, np.pi, 0, 2*np.pi)
            if x>=2*np.pi:
                x_greater_pi=True
                x=x-np.pi

            if x>(np.pi/2):
                y= np.sin(np.tan(x))
            elif 0<x and x<(np.pi/2):
                y= np.cos(-np.tan(x))
            elif (-np.pi/2)<x and x<0:
                y= np.cos(np.tan(x))
            else:
                y= np.sin(-np.tan(x))
            
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
            'cool':my_complex_function,
            'bing':my_generated_function,
            'nice':nice,
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
                            data, "training", color="red")
                    except:
                        pass
                    root.after(10, train_update)
                else:
                    exit_code=trainings_process.exitcode
                    if exit_code == 0:
                        fourier_nn.load_tmp_model()
                        graph.draw_extern_graph_from_func(
                            fourier_nn.predict, "training", color="red", graph_type='crazy')
                    DIE(trainings_process) 
                    trainings_process = None
                    # musik()
                    messagebox.showinfo("training Ended", f"exit code: {exit_code}")

        def train():
            nonlocal fourier_nn
            nonlocal trainings_process

            if trainings_process!=None:
                print('already training')
                return
            
            if hasattr(musik, "out"):
                del musik.out
            if not fourier_nn:
                fourier_nn = FourierNN(graph.export_data())
            else:
                fourier_nn.update_data(graph.export_data())

            # fourier_nn.train_and_plot()
            
            fourier_nn.save_tmp_model()
            trainings_process = multiprocessing.Process(
                target=fourier_nn.train, args=(graph.lst, queue,))
            # process = fourier_nn.train_Process(graph.lst, queue)
            graph.draw_extern_graph_from_data(graph.export_data(), "train_data", color="blue", prio=1)

            root.after(100, trainings_process.start)
            root.after(200, train_update)
            atexit.register(trainings_process.kill)

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

            midi_notes = list(range(60, 72))
            note_durations = [1]*len(midi_notes)

            if fourier_nn:
                notes_list=[]
                if not hasattr(musik, "out"):
                    for i, (note, durration) in enumerate(zip(midi_notes,note_durations)):
                        notes_list.append(fourier_nn.synthesize_2(midi_note=note, duration=durration, sample_rate=44100))
                    musik.out=np.concatenate(notes_list)
                
                sd.play(musik.out, 44100)


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

        # def predTest():
        #     nonlocal fourier_nn
        #     if fourier_nn:
        #         a=[]
        #         for i in range(128):
        #            a.append(fourier_nn.predict(i))
        
        # command= lambda: music.midi_to_musik_live(root, fourier_nn)
        # command= lambda: music.musik_from_file(fourier_nn)
        def clear():
            nonlocal fourier_nn
            if fourier_nn:
                graph.draw_extern_graph_from_func(
                            fourier_nn.predict, "training", color="red")
        button_new_net= tk.Button(root, text='Test', command=clear)
        button_new_net.grid(row=3,column=2, sticky='NSEW')

        def update_2sine():
            pass

        def init():
            # nonlocal fourier_nn
            # if not fourier_nn:
            #     fourier_nn=FourierNN() 
            # utils.process_audio("C:/Users/lucas/Downloads/2-notes-octave-guitar-83275.mp3")
            graph.use_preconfig_drawing(functions['cool'])
            train()
            # command()
            # fourier_nn.load_new_model_from_file("tmp/tan.h5")
            # graph.draw_extern_graph_from_func(fourier_nn.predict, "tan")

        # root.after(500, init)
        # root.after(5*1000, stupid_but_ram)

        # fourier_nn=FourierNN()

        try:
            root.mainloop()
        except KeyboardInterrupt:
            pass

        if trainings_process:
            DIE(trainings_process, 2)
        
        # sys.stdout.exit()

    main()