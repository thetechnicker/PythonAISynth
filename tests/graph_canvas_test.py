# import os
# import unittest
from context import scr
from scr import graph_canvas
import tkinter as tk


if __name__ == "__main__":
    root = tk.Tk()
    graph = graph_canvas.GraphCanvas(root)
    graph.pack(fill=tk.BOTH, expand=True)
    root.mainloop()
