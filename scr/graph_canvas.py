import tkinter as tk
import math

class GraphCanvas(tk.Frame):
    def __init__(self, master):
        super().__init__(master)
        self.canvas = tk.Canvas(self, width=600, height=600, bg='white')
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.canvas.bind('<Configure>', self.maintain_aspect_ratio)
        self.canvas.bind('<Button-1>', self.on_mouse_click)
        self.data = {}
        self.setup_axes()

    def setup_axes(self):
        self.canvas.create_line(0, 300, 600, 300, fill='black')  # X-axis
        self.canvas.create_line(300, 0, 300, 600, fill='black')  # Y-axis
        
        for i in range(-2, 3):
            x = 300 + i * 100
            self.canvas.create_line(x, 290, x, 310, fill='black')
            self.canvas.create_text(x, 320, text=f'{i}π')
        
        for i in range(-1, 2):
            y = 300 - i * 300
            self.canvas.create_line(290, y, 310, y, fill='black')
            self.canvas.create_text(280, y, text=f'{i}')

    def maintain_aspect_ratio(self, event):
        size = min(event.width, event.height)
        self.canvas.config(width=size, height=size)
        self.canvas.delete('all')
        self.setup_axes()
        for x, y in self.data.items():
            self.draw_point(x, y)

    def on_mouse_click(self, event):
        x, y = self.convert_canvas_to_graph_coordinates(event.x, event.y)
        self.data[x] = y
        self.draw_point(x, y)

    def draw_point(self, x, y):
        cx, cy = self.convert_graph_to_canvas_coordinates(x, y)
        self.canvas.create_oval(cx-2, cy-2, cx+2, cy+2, fill='red')

    def export_data(self):
        return self.data

    def convert_canvas_to_graph_coordinates(self, x, y):
        graph_x = (x - 300) / 100 * math.pi
        graph_y = (300 - y) / 300
        return graph_x, graph_y

    def convert_graph_to_canvas_coordinates(self, x, y):
        canvas_x = 300 + x / math.pi * 100
        canvas_y = 300 - y * 300
        return canvas_x, canvas_y

    def draw_bounding_box(self):
        self.canvas.create_rectangle(1, 1, 598, 598, outline='black')

if __name__ == "__main__":
    root = tk.Tk()
    graph = GraphCanvas(root)
    graph.pack(fill=tk.BOTH, expand=True)
    root.mainloop()
