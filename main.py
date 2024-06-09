from threading import Thread
import tkinter as tk

from scr.fourier_neural_network import FourierNN
from scr.graph_canvas import GraphCanvas


def main():
    root = tk.Tk()
    graph = GraphCanvas(root)
    graph.grid()
    fourier_nn:FourierNN=None
    def test():
        nonlocal fourier_nn
        fourier_nn=FourierNN(graph.export_data())
        fourier_nn.train_and_plot()
    def save():
        nonlocal fourier_nn
        print("ok")
        if fourier_nn:
            print("super")
            fourier_nn.convert_to_audio()
        print("ende")
    button= tk.Button(root, text='Train', command=test)
    button2= tk.Button(root, text='musik', command=save)
    button3=tk.Button(root, text='clear', command=graph.clear)
    button.grid()
    button2.grid()
    button3.grid()
    root.mainloop()

if __name__ == "__main__":
    main()