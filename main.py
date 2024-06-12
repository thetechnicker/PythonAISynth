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


if __name__ == "__main__":
    def main():
        root = tk.Tk()
        root.rowconfigure(1, weight=1)
        root.columnconfigure(0, weight=1)
        root.columnconfigure(1, weight=1)
        root.columnconfigure(2, weight=1)
        root.columnconfigure(3, weight=1)
        root.columnconfigure(4, weight=1)
        #root.columnconfigure(5, weight=1)
        #root.columnconfigure(6, weight=1)
        

        list_view_graphs = tk.Listbox(root)
        list_view_graphs.grid(row=0,column=3, rowspan=2, sticky='NSEW')

        list_view_nets = tk.Listbox(root)
        list_view_nets.grid(row=0,column=4, rowspan=2, sticky='NSEW')


        graph = GraphCanvas(root, (900, 300))
        graph.grid(row=1,column=0, columnspan=3, sticky='NSEW')

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

        def train():
            nonlocal fourier_nn
            if not fourier_nn:
                fourier_nn=FourierNN(graph.export_data())
            else:
                fourier_nn.update_data(graph.export_data())
            fourier_nn.train_and_plot()
        
        def musik():
            nonlocal fourier_nn
            if fourier_nn:
                fourier_nn.convert_to_audio()

        def export():
            nonlocal fourier_nn
            if fourier_nn:
                path='./tmp'
                if not os.path.exists(path):
                    os.mkdir(path)
                i=0
                while os.path.exists(path+f"/model{i}.h5"):
                    i+=1
                user_input = simpledialog.askstring("Save Model", "Enter a File Name", initialvalue=f"model{i}", parent=root)
                if not user_input:
                    user_input = f"model{i}"
                file=f"{path}/{user_input}.h5"
                print(file)
                if not os.path.exists(file):
                    try:
                        fourier_nn.save_model(file)
                    except Exception as e:
                        messagebox.showwarning("ERROR - Can't Save Model", f"{e}")
                else:
                    messagebox.showwarning("File already Exists", f"The selected filename {user_input} already exists.")

        def load():
            nonlocal fourier_nn
            filetypes = (('tmodel files', '*.h5'), ('All files', '*.*'))
            filename = filedialog.askopenfilename(title='Open a file', initialdir='.', filetypes=filetypes, parent=root)
            if os.path.exists(filename):
                if not fourier_nn:
                    fourier_nn=FourierNN(data=None)
                fourier_nn.load_new_model_from_file(filename)
                name, color= graph.draw_extern_graph_from_func(fourier_nn.predict, os.path.basename(filename))
                fourier_nn.update_data(graph.get_graph(name))
        
        def create_new_net():
            nonlocal fourier_nn
            if fourier_nn:
                fourier_nn.create_new_model()
                for model in fourier_nn.get_models():
                    print(model.name)

        def add_functions():
            pass

        def invert_function():
            pass

        def select_net():
            pass

        def remove_net():
            pass


        button_train= tk.Button(root, text='Train', command=train)
        button_train.grid(row=2,column=0, sticky='NSEW')

        button_musik= tk.Button(root, text='Musik', command=musik)
        button_musik.grid(row=3,column=0, sticky='NSEW')

        button_clear= tk.Button(root, text='clear', command=graph.clear)
        button_clear.grid(row=2,column=1, sticky='NSEW')

        button_new_net= tk.Button(root, text='create New Net', command=create_new_net)
        button_new_net.grid(row=3,column=1, sticky='NSEW')

        button_export= tk.Button(root, text='export', command=export)
        button_export.grid(row=2,column=2, sticky='NSEW')

        button_load= tk.Button(root, text='load', command=load)
        button_load.grid(row=3,column=2, sticky='NSEW')


        button_add_functions = tk.Button(root, text='add selected functon')
        button_add_functions.grid(row=2,column=3, sticky='NSEW')

        button_invert_function = tk.Button(root, text='invert selected function')
        button_invert_function.grid(row=3,column=3, sticky='NSEW')

        button_select_net = tk.Button(root, text='use selected Net')
        button_select_net.grid(row=2,column=4, sticky='NSEW')
        
        #button_remove_net=None


        graph.draw_extern_graph_from_func(np.sin, "sine", 2, 'black')

        root.mainloop()

    main()
