# import os
# import unittest
from context import src
from src import graph_canvas
import tkinter as tk

"""
Manuel Test for GraphCanvas
"""


if __name__ == "__main__":
    root = tk.Tk()
    graph = graph_canvas.GraphCanvas(root, (900, 300))
    graph.pack(fill=tk.BOTH, expand=True)
    root.mainloop()
