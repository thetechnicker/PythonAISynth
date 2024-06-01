import tkinter as tk
import math

class GraphCanvas(tk.Frame):
    canvas_width=600
    canvas_height=600
    offset=5
    def __init__(self, master):
        super().__init__(master)
        self.canvas = tk.Canvas(self, width=self.canvas_width+self.offset*2, height=self.canvas_height+self.offset*2, bg='white')
        self.canvas.pack(fill=tk.BOTH, expand=True)
        # self.canvas.bind('<Configure>', self.maintain_aspect_ratio)
        # self.canvas.bind('<B1-Motion>', self.on_mouse_move)
        self.data = {}
        self.setup_axes()

    def setup_axes(self):
        self.canvas.create_line(self.offset, self.canvas_height/2, self.canvas_width, self.canvas_height/2, fill='black')  # X-axis
        self.canvas.create_line(self.canvas_width/2, self.offset, self.canvas_width/2, self.canvas_height, fill='black')  # Y-axis
        # self.canvas.create_oval()
        
        for i in range(-2, 3):
            x = (self.canvas_width/2 + i * self.canvas_width/5)+self.offset
            self.canvas.create_line(x, (self.canvas_width/2)-10, x, (self.canvas_width/2)+10, fill='blue')
            self.canvas.create_text(x, (self.canvas_width/2)+10, text=f'{i}π')
        
        for i in range(-1, 2):
            y = (self.canvas_height/2 - i * (self.canvas_height/2))+self.offset
            #print(y)
            self.canvas.create_line((self.canvas_height/2)-10, y, (self.canvas_height/2)+10, y, fill='red')
            #self.canvas.create_oval((self.canvas_height/2)-10, y+10, (self.canvas_height/2)+10, y-10,  fill='black')
            #self.canvas.create_text((self.canvas_height/2)-10, y, text=f'{i}')

    def maintain_aspect_ratio(self, event):
        size = min(event.width, event.height)
        self.canvas.config(width=size, height=size)
        self.canvas.delete('all')
        self.setup_axes()
        for x, y in self.data.items():
            self.draw_point(x, y)

    def on_mouse_move(self, event):
        x, y = self.convert_canvas_to_graph_coordinates(event.x, event.y)
        if -1<=y<=1 and -math.pi < x < math.pi:
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
