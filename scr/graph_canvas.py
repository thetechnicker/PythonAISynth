from copy import copy
import tkinter as tk
import math

import numpy as np

from scr import utils

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
    canvas_width = 600
    canvas_height = 600
    aspect_ratio = 1
    offset = 10
    lower_end= -math.pi
    upper_end= math.pi

    def __init__(self, master, size:tuple[int, int]=None):
        if size:
            self.canvas_width = size[0]
            self.canvas_height = size[1]
            self.aspect_ratio=self.canvas_width/self.canvas_height
        super().__init__(master)
        self.canvas = tk.Canvas(self, width=self.canvas_width+self.offset*2,
                                height=self.canvas_height+self.offset*2, bg='white')
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.canvas.bind('<Configure>', self.resize)
        
        self.canvas.bind('<B1-Motion>', self.on_mouse_move_interpolate) # alternative functions: on_mouse_move, on_mouse_move_optimized
        self.canvas.bind('<ButtonRelease-1>', self.motion_end)
        self.canvas.bind('<space>', self.on_space_press)
        self.data: list[tuple[float, float]] = []
        self.setup_axes()
        self.lst=np.linspace(-np.pi, np.pi, 200)

    def motion_end(self, event):
        if hasattr(self, 'old_point'):
            del self.old_point

    def on_space_press(self, event):
        self.on_mouse_move(event)

    def setup_axes(self):
        self.canvas.create_line(self.offset, self.canvas_height/2+self.offset, self.canvas_width +
                                self.offset, self.canvas_height/2+self.offset, fill='black')  # X-axis
        self.canvas.create_line(self.canvas_width/2+self.offset, self.offset, self.canvas_width /
                                2+self.offset, self.canvas_height+self.offset, fill='black')  # Y-axis
        # self.canvas.create_oval()

        for i in range(-1, 2):
            x = (self.canvas_width/2 + i * self.canvas_width/2)+self.offset
            self.canvas.create_line(x, (self.canvas_height/2)-10+self.offset,
                                    x, (self.canvas_height/2)+10+self.offset, fill='blue')
            self.canvas.create_text(
                x, (self.canvas_height/2)+10+self.offset, text=f'{i}π')

        for i in range(-1, 2):
            y = (self.canvas_height/2 - i * (self.canvas_height/2))+self.offset
            # print(y)
            self.canvas.create_line((self.canvas_width/2)-10+self.offset,
                                    y, (self.canvas_width/2)+10+self.offset, y, fill='red')
            # self.canvas.create_oval((self.canvas_height/2)-10, y+10, (self.canvas_height/2)+10, y-10,  fill='black')
            self.canvas.create_text((self.canvas_width/2)-10, y, text=f'{i}')

    first = True

    def resize(self, event):
        if self.first:
            self.first = False
            return
        self.canvas_width = event.width-self.offset*2
        self.canvas_height = event.height-self.offset*2
        self.canvas.config(width=event.width, height=event.height)
        self.canvas.delete('all')
        self.setup_axes()
        for x, y in self.data:
            self.draw_point(x, y)
        # self.first=True # Keeping this, if it breaks again.
    
    # def maintain_aspect_ratio(self, event):
    #     """Not Working"""
    #     if self.first:
    #         self.first = False
    #         return
    #     size = min(event.width, event.height)

    #     width = size * self.aspect_ratio
    #     height = size * (1/self.aspect_ratio)

    #     self.canvas_width = width-self.offset*2
    #     self.canvas_height = height-self.offset*2

    #     self.canvas.config(width=width, height=height)
    #     self.canvas.delete('all')
    #     self.setup_axes()
    #     for x, y in self.data:
    #         self.draw_point(x, y)

    def on_mouse_move_interpolate(self, event):
        new_point = (event.x, event.y)
        if hasattr(self, 'old_point'):
            old_point = self.old_point
            dist = abs(new_point[0] - old_point[0])
            if dist > 2:
                t_values = np.arange(dist) / dist
                points = utils.interpolate_vectorized(old_point, new_point, t_values)
                for point in points:
                    interpolate_event = copy(event)
                    interpolate_event.x, interpolate_event.y = point
                    self.on_mouse_move_optimized(interpolate_event)
            else:
                self.on_mouse_move_optimized(event)
        else:
            self.on_mouse_move_optimized(event)

        self.canvas.delete('all')
        self.setup_axes()
        for x, y in self.data:
            self.draw_point(x, y)
        self.old_point = new_point

    def on_mouse_move(self, event):
        x, y = self.convert_canvas_to_graph_coordinates(event.x, event.y)
        x = min(self.lst, key=lambda _y: abs(x - _y))
        
        if -1 <= y <= 1 and self.lower_end < x < self.upper_end:
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
            for x, y in self.data:
                self.draw_point(x, y)

    def on_mouse_move_optimized(self, event):
        x, y = self.convert_canvas_to_graph_coordinates_optimized(event.x, event.y)
        x = min(self.lst, key=lambda _y: abs(x - _y))

        if -1 <= y <= 1 and self.lower_end < x < self.upper_end:
            data_dict = dict(self.data)
            data_dict[x] = y
            self.data = list(data_dict.items())


    def draw_point(self, x, y):
        cx, cy = self.convert_graph_to_canvas_coordinates_optimized(x, y)
        self.canvas.create_oval(cx-2, cy-2, cx+2, cy+2, fill='red')

    def export_data(self):
        return copy(self.data)

    def convert_canvas_to_graph_coordinates(self, x, y):
        ax_1, ax_2, bx_1, bx_2 = self.offset, self.canvas_width, self.upper_end, self.lower_end
        ay_1, ay_2, by_1, by_2 = self.offset, self.canvas_height, -1, 1
        graph_x = map_value(x, ax_1, ax_2, bx_1, bx_2)
        graph_y = map_value(y, ay_1, ay_2, by_1, by_2)
        return graph_x, graph_y

    def convert_graph_to_canvas_coordinates(self, x, y):
        ax_1, ax_2, bx_1, bx_2 = self.upper_end, self.lower_end, self.offset, self.canvas_width
        ay_1, ay_2, by_1, by_2 = -1, 1, self.offset, self.canvas_height
        graph_x = map_value(x, ax_1, ax_2, bx_1, bx_2)
        graph_y = map_value(y, ay_1, ay_2, by_1, by_2)
        return graph_x, graph_y

    def convert_canvas_to_graph_coordinates_optimized(self, x, y):
        graph_x = self.map_value(x, self.offset, self.canvas_width, self.upper_end, self.lower_end)
        graph_y = self.map_value(y, self.offset, self.canvas_height, -1, 1)
        return graph_x, graph_y

    def convert_graph_to_canvas_coordinates_optimized(self, x, y):
        graph_x = self.map_value(x, self.upper_end, self.lower_end, self.offset, self.canvas_width)
        graph_y = self.map_value(y, -1, 1, self.offset, self.canvas_height)
        return graph_x, graph_y

    def map_value(self, value, leftMin, leftMax, rightMin, rightMax):
        # Assuming map_value function looks something like this
        leftSpan = leftMax - leftMin
        rightSpan = rightMax - rightMin
        valueScaled = float(value - leftMin) / float(leftSpan)
        return rightMin + (valueScaled * rightSpan)

    def draw_bounding_box(self):
        self.canvas.create_rectangle(1, 1, 598, 598, outline='black')


# moved to ./tests
# if __name__ == "__main__":
#     root = tk.Tk()
#     graph = GraphCanvas(root)
#     graph.pack(fill=tk.BOTH, expand=True)
#     root.mainloop()
