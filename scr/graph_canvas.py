import tkinter as tk
import math


def map_value(x, a1, a2, b1, b2):
    """
    Maps a value from one range to another.

    Parameters:
    x (float): The value to map.
    a1 (float): The lower bound of the original range.
    a2 (float): The upper bound of the original range.
    b1 (float): The lower bound of the target range.
    b2 (float): The upper bound of the target range.

    Returns:
    float: The mapped value in the target range.
    """
    return b1 + ((x - a1) * (b2 - b1)) / (a2 - a1)


class GraphCanvas(tk.Frame):
    canvas_width = 600
    canvas_height = 600
    offset = 5

    def __init__(self, master):
        super().__init__(master)
        self.canvas = tk.Canvas(self, width=self.canvas_width+self.offset*2,
                                height=self.canvas_height+self.offset*2, bg='white')
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.canvas.bind('<Configure>', self.maintain_aspect_ratio)
        self.canvas.bind('<B1-Motion>', self.on_mouse_move)
        self.canvas.bind('<space>', self.on_space_press)
        self.data: list[tuple[float, float]] = []
        self.setup_axes()

    def on_space_press(self, event):
        self.on_mouse_move(event)

    def setup_axes(self):
        self.canvas.create_line(self.offset, self.canvas_height/2+self.offset, self.canvas_width +
                                self.offset, self.canvas_height/2+self.offset, fill='black')  # X-axis
        self.canvas.create_line(self.canvas_width/2+self.offset, self.offset, self.canvas_width /
                                2+self.offset, self.canvas_height+self.offset, fill='black')  # Y-axis
        # self.canvas.create_oval()

        for i in range(-2, 3):
            x = (self.canvas_width/2 + i * self.canvas_width/5)+self.offset
            self.canvas.create_line(x, (self.canvas_width/2)-10+self.offset,
                                    x, (self.canvas_width/2)+10+self.offset, fill='blue')
            self.canvas.create_text(
                x, (self.canvas_width/2)+10+self.offset, text=f'{i}π')

        for i in range(-1, 2):
            y = (self.canvas_height/2 - i * (self.canvas_height/2))+self.offset
            # print(y)
            self.canvas.create_line((self.canvas_height/2)-10+self.offset,
                                    y, (self.canvas_height/2)+10+self.offset, y, fill='red')
            # self.canvas.create_oval((self.canvas_height/2)-10, y+10, (self.canvas_height/2)+10, y-10,  fill='black')
            self.canvas.create_text((self.canvas_height/2)-10, y, text=f'{i}')

    first = True

    def maintain_aspect_ratio(self, event):
        if self.first:
            self.first = False
            return
        size = min(event.width, event.height)
        self.canvas_width = size
        self.canvas_height = size
        self.canvas.config(width=size+self.offset*2, height=size+self.offset*2)
        self.canvas.delete('all')
        self.setup_axes()
        for x, y in self.data:
            self.draw_point(x, y)

    def on_mouse_move(self, event):
        x, y = self.convert_canvas_to_graph_coordinates(event.x, event.y)
        x = round(x, 1)
        if -1 <= y <= 1 and -2*math.pi < x < 2*math.pi:
            if self.data:
                if any(x == _data for _data, _ in self.data):
                    index = [i for i, data in enumerate(
                        self.data) if x == (data[0])][0]
                    self.data[index] = (x, y,)
                else:
                    self.data.append((x, y))
            else:
                self.data.append((x, y))
            self.canvas.delete('all')
            self.setup_axes()
            for i, (x, y) in enumerate(self.data):
                # print(i,x,y)
                self.draw_point(x, y)

    def draw_point(self, x, y):
        cx, cy = self.convert_graph_to_canvas_coordinates(x, y)
        self.canvas.create_oval(cx-2, cy-2, cx+2, cy+2, fill='red')

    def export_data(self):
        return self.data

    def convert_canvas_to_graph_coordinates(self, x, y):
        ax_1, ax_2, bx_1, bx_2 = self.offset, self.canvas_width, -2*math.pi, 2*math.pi
        ay_1, ay_2, by_1, by_2 = self.offset, self.canvas_height, -1, 1
        graph_x = map_value(x, ax_1, ax_2, bx_1, bx_2)
        graph_y = map_value(y, ay_1, ay_2, by_1, by_2)
        return graph_x, graph_y


    def convert_graph_to_canvas_coordinates(self, x, y):
        ax_1, ax_2, bx_1, bx_2 = -2*math.pi, 2*math.pi, self.offset, self.canvas_width
        ay_1, ay_2, by_1, by_2 = -1, 1, self.offset, self.canvas_height
        graph_x = map_value(x, ax_1, ax_2, bx_1, bx_2)
        graph_y = map_value(y, ay_1, ay_2, by_1, by_2)
        return graph_x, graph_y
    
    def draw_bounding_box(self):
        self.canvas.create_rectangle(1, 1, 598, 598, outline='black')


if __name__ == "__main__":
    root = tk.Tk()
    graph = GraphCanvas(root)
    graph.pack(fill=tk.BOTH, expand=True)
    root.mainloop()
