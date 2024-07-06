from copy import copy
import time
import tkinter as tk
import math
from typing import Literal

import numpy as np

from scr import utils

class GraphCanvas(tk.Frame):
    LEVEL_OF_DETAIL=250
    INCLUDE_0 = False

    canvas_width = 600
    canvas_height = 600
    aspect_ratio = 1
    offset = 10
    point_radius=5

    lower_end_x= -math.pi
    upper_end_x= math.pi
    lower_end_y = 1
    upper_end_y = -1

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
        
        self.canvas.bind('<B1-Motion>', self.on_mouse_move_interpolate)
        self.canvas.bind('<ButtonRelease-1>', self.motion_end)
        self.canvas.bind('<space>', self.on_space_press)
        self.lst=np.linspace(-np.pi, np.pi, self.LEVEL_OF_DETAIL)
        if 0 not in self.lst and self.INCLUDE_0:
            self.lst=np.append(self.lst, 0)
        self.lst=np.sort(self.lst)
        self.data: list[tuple[float, float]] = []
        self.clear()

    def _draw(self):
        #timestamp=time.perf_counter_ns()
        self.canvas.delete('all')
        self.setup_axes()
        for x, y in self.data:
            self.draw_point(x, y)
        if hasattr(self, 'extern_graph'):
            self._draw_extern_graph()
        #print(f"Time taken: {(time.perf_counter_ns()-timestamp)/1_000_000_000}s")

    def motion_end(self, event):
        if hasattr(self, 'old_point'):
            del self.old_point

    
    def clear(self):
        self.data = [(x, 0) for x in self.lst]
        self._draw()

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
        self._draw()
        self.first=True # Keeping this, if it breaks again.
    
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
                    self.eval_mouse_move_event(interpolate_event)
            else:
                self.eval_mouse_move_event(event)
        else:
            self.eval_mouse_move_event(event)

        self._draw()
        self.old_point = new_point

    def eval_mouse_move_event(self, event):
        x, y = self.convert_canvas_to_graph_coordinates_optimized(event.x, event.y)
        x = min(self.lst, key=lambda _y: abs(x - _y))

        if  utils.is_in_interval(y, self.lower_end_y, self.upper_end_y) and utils.is_in_interval(x, self.lower_end_x, self.upper_end_x):
            data_dict = dict(self.data)
            data_dict[x] = y
            self.data = list(data_dict.items())


    def draw_point(self, x, y):
        cx, cy = self.convert_graph_to_canvas_coordinates_optimized(x, y)
        self.canvas.create_oval(cx-self.point_radius, cy-self.point_radius, cx+self.point_radius, cy+self.point_radius, fill='red')

    def export_data(self):
        return copy(self.data)


    def convert_canvas_to_graph_coordinates_optimized(self, x, y):
        graph_x = utils.map_value(x-self.offset, 0, self.canvas_width, self.lower_end_x, self.upper_end_x)
        graph_y = utils.map_value(y-self.offset, 0, self.canvas_height, self.lower_end_y, self.upper_end_y)
        return graph_x, graph_y

    def convert_graph_to_canvas_coordinates_optimized(self, x, y):
        graph_x = utils.map_value(x, self.lower_end_x, self.upper_end_x, 0, self.canvas_width)+self.offset
        graph_y = utils.map_value(y, self.lower_end_y, self.upper_end_y, 0, self.canvas_height)+self.offset
        return graph_x, graph_y

    def draw_bounding_box(self):
        self.canvas.create_rectangle(1, 1, 598, 598, outline='black')

    def _eval_func(self, function, x):
        val=function(x)
        return val if utils.is_in_interval(val, -1, 1) else 0

    def use_preconfig_drawing(self, function):
        self.data = [(x, self._eval_func(function, x)) for x in self.lst]
        self._draw()

    def get_graph(self, name):
        return self.extern_graph[name]

    def _draw_extern_graph(self):
        legend_x = 10  # The x-coordinate of the top-left corner of the legend
        legend_y = 10  # The y-coordinate of the top-left corner of the legend
        legend_spacing = 20  # The vertical spacing between items in the legend
    
        for i, (name, (graph, color, width, graph_type)) in enumerate(self.extern_graph.items()):
            # Draw a small line of the same color as the graph
            self.canvas.create_line(legend_x, legend_y + i * legend_spacing,
                                    legend_x + 20, legend_y + i * legend_spacing,
                                    fill=color, width=2)
            # Draw the name of the graph
            self.canvas.create_text(legend_x + 30, legend_y + i * legend_spacing,
                                    text=name, anchor='w')
            if graph_type:# and False:
                for a, b in utils.pair_iterator(graph):
                    a_new=self.convert_graph_to_canvas_coordinates_optimized(*a)
                    b_new=self.convert_graph_to_canvas_coordinates_optimized(*b)
                    self.canvas.create_line(*a_new,*b_new, width=width, fill=color)
            else:
                for a in graph:
                    x,y=self.convert_graph_to_canvas_coordinates_optimized(*a)
                    self.canvas.create_oval(x-width/2, y-width/2, x+width/2, y+width/2)
    
    def draw_extern_graph_from_func(self, function, name=None, width=None, color=None, graph_type='line'):
        if not width:
            width=self.point_radius/2
        if not color:
            color=utils.get_prepared_random_color()
        if not name:
            name = f"funciton {len(list(self.extern_graph.keys()))}"
        try:
            x=self.lst #
            if graph_type=="cracy":
                x=np.linspace(self.lower_end_x, self.upper_end_x, 44100)
                graph_type="line"
            data =  list(zip(x,function(x)))
            if not hasattr(self, 'extern_graph'):
                self.extern_graph={}
            self.extern_graph[name] = (data,color,width,graph_type)
            self._draw()
        except Exception as e:
            print(e)

    def draw_extern_graph_from_data(self, data, name=None, width=None, color=None, graph_type='line'):
        if not width:
            width=self.point_radius/2
        if not color:
            color=utils.get_prepared_random_color()
        if not name:
            name = f"funciton {len(list(self.extern_graph.keys()))}"
        try:
            #data =  list(zip(self.lst,data))
            if not hasattr(self, 'extern_graph'):
                self.extern_graph={}
            self.extern_graph[name] = (data,color,width,graph_type)
            self._draw()
        except Exception as e:
            print(e)
        

# moved to ./tests
# if __name__ == "__main__":
#     root = tk.Tk()
#     graph = GraphCanvas(root)
#     graph.pack(fill=tk.BOTH, expand=True)
#     root.mainloop()
 