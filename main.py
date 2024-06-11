import os
from threading import Thread
import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
from tensorflow import keras

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
        

        graph = GraphCanvas(root, (900, 300))
        graph.grid(row=1,column=0, columnspan=5, sticky='NSEW')

        functions = {
            'relu':keras.activations.relu,
            'elu':keras.activations.elu,
            'linear':keras.activations.linear,
            'sigmoid':keras.activations.sigmoid,
            'exponential':keras.activations.exponential,
            'hard_sigmoid':keras.activations.hard_sigmoid,
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

        def test():
            nonlocal fourier_nn
            fourier_nn=FourierNN(graph.export_data())
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
                file=f"{path}/model{i}.h5"
                fourier_nn.save_model(file)

        def load():
            nonlocal fourier_nn
            filetypes = (('tmodel files', '*.h5'), ('All files', '*.*'))
            filename = filedialog.askopenfilename(title='Open a file', initialdir='.', filetypes=filetypes, parent=root)
            if os.path.exists(filename):
                if not fourier_nn:
                    fourier_nn=FourierNN(data=None)
                fourier_nn.load_model(filename)
                graph.draw_extern_graph_from_func(os.path.basename(filename), fourier_nn.predict)
        

        button= tk.Button(root, text='Train', command=test)
        button.grid(row=2,column=0, sticky='NSEW')

        button2= tk.Button(root, text='Musik', command=musik)
        button2.grid(row=2,column=1, sticky='NSEW')

        button3= tk.Button(root, text='clear', command=graph.clear)
        button3.grid(row=2,column=2, sticky='NSEW')

        button4= tk.Button(root, text='export', command=export)
        button4.grid(row=2,column=3, sticky='NSEW')

        button5= tk.Button(root, text='load', command=load)
        button5.grid(row=2,column=4, sticky='NSEW')

        root.mainloop()

    main()
