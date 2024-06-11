from threading import Thread
import tkinter as tk

from scr.fourier_neural_network import FourierNN
from scr.graph_canvas import GraphCanvas


if __name__ == "__main__":
    def main():
        root = tk.Tk()
        root.rowconfigure(0, weight=1)
        root.columnconfigure(0, weight=1)
        root.columnconfigure(1, weight=1)
        root.columnconfigure(2, weight=1)
        graph = GraphCanvas(root)
        graph.grid(row=0,column=0, columnspan=3, sticky='NSEW')
        fourier_nn=None
        def test():
            nonlocal fourier_nn
            fourier_nn=FourierNN(graph.export_data())
            fourier_nn.train_and_plot()
        
        def musik():
            nonlocal fourier_nn
            fourier_nn.convert_to_audio()

        button= tk.Button(root, text='Train', command=test)
        button.grid(row=1,column=0, sticky='NSEW')

        button2= tk.Button(root, text='Musik', command=musik)
        button2.grid(row=1,column=1, sticky='NSEW')

        button3= tk.Button(root, text='clear', command=graph.clear)
        button3.grid(row=1,column=2, sticky='NSEW')

        root.mainloop()

    main()
