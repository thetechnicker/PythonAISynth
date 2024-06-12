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
        root = tk.Tk()
        root.rowconfigure(1, weight=1)
        root.columnconfigure(0, weight=1)
        root.columnconfigure(1, weight=1)
        root.columnconfigure(2, weight=1)
        root.columnconfigure(3, weight=1)
        root.columnconfigure(4, weight=1)
        

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
        label.grid(row=0,column=0, sticky='NSEW')

        predefined_functions_select = ttk.Combobox(root, textvariable=n, state="readonly")
        predefined_functions_select.bind('<<ComboboxSelected>>', predefined_functions)
        predefined_functions_select.grid(row=0,column=1, columnspan=2, sticky='NSEW')
        predefined_functions_select['values'] = list(functions.keys())


        fourier_nn=None

        def train():
            nonlocal fourier_nn
            if not fourier_nn:
                fourier_nn=FourierNN(graph.export_data())
            else:
                fourier_nn.update_data(graph.export_data())
            fourier_nn.train_and_plot()
            update_net_list()
        
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
            filetypes = (('HDF5 files', '*.h5'), ('Keras files', '*.keras'), ('All files', '*.*'))
            filename = filedialog.askopenfilename(title='Open a file', initialdir='.', filetypes=filetypes, parent=root)
            if os.path.exists(filename):
                if not fourier_nn:
                    fourier_nn=FourierNN(data=None)
                fourier_nn.load_new_model_from_file(filename)
                name, color= graph.draw_extern_graph_from_func(fourier_nn.predict, os.path.basename(filename).split('.')[0])
                #list_view_graphs.insert(tk.END, f"{name}")
                fourier_nn.update_data(data=graph.get_graph(name=name)[0])
            update_net_list()
            update_function_list()
        
        def create_new_net():
            nonlocal fourier_nn
            if fourier_nn:
                fourier_nn.create_new_model()
                update_net_list()


        def update_function_list():
            list_view_graphs.delete(0, tk.END)
            graph_names=graph.get_graph_names()
            for i, name in enumerate(graph_names):
                list_view_graphs.insert(tk.END, f"{name}")
                color=graph.get_graph(name)[1]
                list_view_graphs.itemconfig(i, {'fg':color})

        def update_net_list():
            nonlocal fourier_nn
            if fourier_nn:
                list_view_nets.delete(0, tk.END)
                for i, model in enumerate(fourier_nn.get_models()):
                    current="(current)" if i==0 else ""
                    list_view_nets.insert(tk.END, f"{model.name}{current}:{i}")

        def add_functions():
            pass

        def invert_function():
            pass

        def select_net():
            nonlocal fourier_nn
            if fourier_nn:
                net_name=list_view_nets.get(list_view_nets.curselection())
                fourier_nn.change_model(int(net_name.split(':')[1]))
                graph.use_preconfig_drawing_parallel(fourier_nn.predict)
                update_net_list()

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


        button_add_functions = tk.Button(root, text='add selected functon', command=add_functions)
        button_add_functions.grid(row=2,column=3, sticky='NSEW')

        button_invert_function = tk.Button(root, text='invert selected function', command=invert_function)
        button_invert_function.grid(row=3,column=3, sticky='NSEW')

        button_select_net = tk.Button(root, text='use selected Net', command=select_net)
        button_select_net.grid(row=2,column=4, sticky='NSEW')
        
        #button_remove_net=None


        graph.draw_extern_graph_from_func(np.sin, "base function", width=2, color='black')
        update_function_list()
        update_net_list()
        root.mainloop()

    main()
