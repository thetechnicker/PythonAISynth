import atexit
from multiprocessing import Queue
import multiprocessing
import os
from threading import Thread
import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
from tkinter import simpledialog
from tkinter import messagebox
from tensorflow import keras
import numpy as np

from scr.fourier_neural_network import FourierNN
from scr.graph_canvas import GraphCanvas
from scr.simple_input_dialog import askStringAndSelectionDialog


if __name__ == "__main__":
    def main():
        process=None
        queue=Queue(-1)
        root = tk.Tk()
        root.rowconfigure(1, weight=1)
        root.columnconfigure(0, weight=1)
        root.columnconfigure(1, weight=1)
        root.columnconfigure(2, weight=1)
        root.columnconfigure(3, weight=1)
        root.columnconfigure(4, weight=1)
        

        graph = GraphCanvas(root, (900, 300))
        graph.grid(row=1,column=0, columnspan=5, sticky='NSEW')

        functions = {
            'sin': np.sin,
            'cos': np.cos,
            'tan': np.tan,
            'relu':keras.activations.relu,
            'elu':keras.activations.elu,
            'linear':keras.activations.linear,
            'sigmoid':keras.activations.sigmoid,
            'exponential':keras.activations.exponential,
            'selu':keras.activations.selu,
            'gelu':keras.activations.gelu
        }

        n = tk.StringVar()
        def predefined_functions(event):
            graph.use_preconfig_drawing(functions[n.get()])
    
        label=tk.Label(root, text='Predifined functions')
        label.grid(row=0,column=0, columnspan=2, sticky='NSEW')

        predefined_functions_select = ttk.Combobox(root, textvariable=n)
        predefined_functions_select.bind('<<ComboboxSelected>>', predefined_functions)
        predefined_functions_select.grid(row=0,column=2, columnspan=3, sticky='NSEW')
        predefined_functions_select['values'] = list(functions.keys())


        fourier_nn=None

        def train_update():
            if process.is_alive():
                try:
                    data=queue.get_nowait()
                    data=data.reshape(-1)
                    graph.data=list(zip(graph.lst, data))
                    graph._draw()
                except:
                    pass
                root.after(500, train_update)

        def proc_exit():
            nonlocal process
            if process:
                process.kill()

        def train():
            nonlocal fourier_nn
            nonlocal process
            if process:
                return
            if not fourier_nn:
                 fourier_nn=FourierNN(graph.export_data())
            process=multiprocessing.Process(target=fourier_nn.train, args=(graph.lst, queue, True))
            process.start()
            graph.draw_extern_graph_from_data(graph.data, "orig_data")
            print(process.pid)
            root.after(10, train_update)
            atexit.register(proc_exit)
        
        def musik():
            nonlocal fourier_nn
            if fourier_nn:
                fourier_nn.convert_to_audio()

        def export():
            default_format='keras'
            nonlocal fourier_nn
            if fourier_nn:
                path='./tmp'
                if not os.path.exists(path):
                    os.mkdir(path)
                i=0
                while os.path.exists(path+f"/model{i}.{default_format}"):
                    i+=1

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

                file=f"{path}/{name}.{file_format}"
                print(file)
                if not os.path.exists(file):
                    try:
                        fourier_nn.save_model(file)
                    except Exception as e:
                        messagebox.showwarning("ERROR - Can't Save Model", f"{e}")
                else:
                    messagebox.showwarning("File already Exists", f"The selected filename {name} already exists.")


        def load():
            nonlocal fourier_nn
            filetypes = (('Keras files', '*.keras'), ('HDF5 files', '*.h5'), ('All files', '*.*'))
            filename = filedialog.askopenfilename(title='Open a file', initialdir='.', filetypes=filetypes, parent=root)
            if os.path.exists(filename):
                if not fourier_nn:
                    fourier_nn=FourierNN(data=None)
                fourier_nn.load_new_model_from_file(filename)
                name = os.path.basename(filename).split('.')[0]
                graph.draw_extern_graph_from_func(fourier_nn.predict, name)
                print(name)
                fourier_nn.update_data(data=graph.get_graph(name=name)[0])
        

        button= tk.Button(root, text='Train', command=train)
        button.grid(row=2,column=0, sticky='NSEW')

        button2= tk.Button(root, text='Musik', command=musik)
        button2.grid(row=2,column=1, sticky='NSEW')

        button3= tk.Button(root, text='clear', command=graph.clear)
        button3.grid(row=2,column=2, sticky='NSEW')

        button4= tk.Button(root, text='export', command=export)
        button4.grid(row=2,column=3, sticky='NSEW')

        button5= tk.Button(root, text='load', command=load)
        button5.grid(row=2,column=4, sticky='NSEW')

        def init():
            graph.use_preconfig_drawing(functions['tan'])
            train()

        root.after(500, init)

        try:
            root.mainloop()
        except KeyboardInterrupt:
            if process:
                process.join(5)
                while process.is_alive():
                    process.kill()

    main()
