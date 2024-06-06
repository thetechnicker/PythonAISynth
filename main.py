from threading import Thread
import tkinter as tk

from scr.fourier_neural_network import FourierNN
from scr.graph_canvas import GraphCanvas


if __name__ == "__main__":
    root = tk.Tk()
    graph = GraphCanvas(root)
    graph.pack(fill=tk.BOTH, expand=True)
    def test():
        fourier_nn=FourierNN(graph.export_data())
        fourier_nn.train_and_plot()
    button= tk.Button(root, text='Train', command=test)
    button.pack()
    root.mainloop()
