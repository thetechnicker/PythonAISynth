import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.widgets import Cursor
import tkinter as tk
from tkinter import ttk

from scr.utils import run_after_ms, tk_after_errorless

matplotlib.use('TkAgg')


class GraphCanvas(ttk.Frame):
    def __init__(self, master, *args, **kwargs):
        super().__init__(master)
        self.master = master

        # Define the range for x from 0 to 2π with 250 evenly spaced points
        self.x = np.linspace(0, 2 * np.pi, 250)
        self.y = np.zeros_like(self.x)  # Initial y values set to 0

        plt.style.use('dark_background')
        # Create the plot
        self.fig, self.ax = plt.subplots(figsize=(10, 6))
        self.line, = self.ax.plot(
            self.x, self.y, '-o', markersize=5)  # Red dots

        # Set the x and y axis limits
        self.ax.set_xlim(0, 2 * np.pi)
        self.ax.set_ylim(-2, 2)

        # Set the x and y axis labels
        self.ax.set_xlabel('x (radians)')
        self.ax.set_ylabel('y')

        # Customize the ticks on the x-axis
        self.ax.set_xticks([0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi])
        self.ax.set_xticklabels(['0', 'π/2', 'π', '3π/2', '2π'])

        # Draw the grid
        self.ax.grid()

        # Add a title
        self.ax.set_title('Move the Closest Dot Along the Y-Axis')

        # Create a cursor for better visibility
        self.cursor = Cursor(self.ax, useblit=True, color='blue', linewidth=1)

        # Variables to track the selected point
        self.selected_index = None
        self.last_index = None

        # Dictionary to track existing plots by name
        self.existing_plots = {}

        # Connect the events to the functions
        self.fig.canvas.mpl_connect('button_press_event', self.on_mouse_press)
        self.fig.canvas.mpl_connect(
            'button_release_event', self.on_mouse_release)
        self.fig.canvas.mpl_connect(
            'motion_notify_event', self.on_mouse_motion)

        # Create a canvas to embed the plot in Tkinter
        self.redraw_needed = False
        tk_after_errorless(self, 100, self.draw_idle)

        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def draw_idle(self):
        if self.redraw_needed:
            self.canvas.draw_idle()
            tk_after_errorless(self, 10, self.draw_idle)
        else:
            tk_after_errorless(self, 100, self.draw_idle)

    def find_closest_point(self, x_mouse):
        distances = np.sqrt((self.x - x_mouse) ** 2)
        return np.argmin(distances)

    def update_y(self, event):
        if event.inaxes == self.ax and self.selected_index is not None:
            y_new = event.ydata
            self.y[self.selected_index] = y_new
            self.line.set_ydata(self.y)
            self.canvas.draw_idle()

    def on_mouse_press(self, event):
        if event.inaxes == self.ax and event.button == 1:  # Left mouse button
            self.selected_index = self.find_closest_point(event.xdata)
            self.last_index = self.selected_index
            self.y[self.selected_index] = event.ydata
            self.line.set_ydata(self.y)
            self.redraw_needed = True

    def on_mouse_motion(self, event):
        if event.inaxes == self.ax and self.selected_index is not None:
            new_index = self.find_closest_point(event.xdata)
            if new_index != self.selected_index:
                if self.last_index is not None:
                    start_index = self.last_index
                    end_index = new_index
                    indices = np.arange(
                        start_index, end_index + 1) if new_index > self.selected_index else np.arange(start_index, end_index - 1, -1)
                    percentages = (indices - start_index) / \
                        (end_index - start_index)
                    self.y[indices] = event.ydata * percentages + \
                        self.y[start_index] * (1 - percentages)
                self.selected_index = new_index
                self.line.set_ydata(self.y)
                self.last_index = new_index
            self.redraw_needed = True

    def on_mouse_release(self, event):
        self.last_index = self.selected_index
        self.selected_index = None
        self.redraw_needed = False

    def destroy(self):
        plt.close(self.fig)
        super().destroy()

    def plot_points(self, points, name=None, type="points"):
        """Plot multiple points on the graph.

        Args:
            points (list of tuples): List of (x, y) tuples to plot.
            name (str, optional): Name for the plot legend.
            type (str): Type of plot - "points" for individual points, "line" for connecting points.
        """
        # Unzip the list of tuples into x and y values
        x_values, y_values = zip(*points)

        # Check if a plot with the same name already exists
        if name in self.existing_plots:
            # Remove the existing plot
            self.existing_plots[name].remove()

        if type == "points":
            plot_line, = self.ax.plot(
                x_values, y_values, 'ro', label=name)  # 'ro' for red dots
        elif type == "line":
            plot_line, = self.ax.plot(
                x_values, y_values, 'r-', label=name)  # 'r-' for red line
        else:
            raise ValueError("Invalid type. Use 'points' or 'line'.")

        if name:
            self.ax.legend()
            self.existing_plots[name] = plot_line  # Store the new plot line

        self.canvas.draw_idle()

    def plot_function(self, func, x_range=None, overwrite=False, name=None):
        """Plot a function over a specified range or use self.x if no range is provided.
        If overwrite is True, update the existing y-values of the red dots.
        If name is provided, update the plot with that name if it exists."""
        if x_range is None:
            x_values = self.x  # Use the existing x values
        else:
            num = 250
            if len(x_range) == 3:
                num = x_range[2]
            x_values = np.linspace(x_range[0], x_range[1], num)

        y_values = func(x_values)  # np.clip(, -1, 1)

        if name:
            # Check if a plot with the specified name already exists
            if name in self.existing_plots:
                # Update the y-values of the existing plot
                self.existing_plots[name].set_ydata(y_values)
            else:
                # Create a new plot for the function
                plot_line, = self.ax.plot(
                    x_values, y_values, label=f'Function: {name}')
                self.ax.legend()
                # Store the new plot line
                self.existing_plots[name] = plot_line
        else:
            if overwrite:
                # Update the y-values of the existing red dots
                self.y = y_values
                self.line.set_ydata(self.y)
            else:
                # Check if a plot with the same name already exists
                if func.__name__ in self.existing_plots:
                    # Remove the existing plot
                    self.existing_plots[func.__name__].remove()

                # Create a new plot for the function
                plot_line, = self.ax.plot(
                    x_values, y_values, label=f'Function: {func.__name__}')
                self.ax.legend()
                # Store the new plot line
                self.existing_plots[func.__name__] = plot_line

        self.canvas.draw_idle()

    def export_data(self):
        return list(zip(self.x, self.y))

    def clear(self):
        self.y = np.zeros_like(self.x)
        self.line.set_ydata(self.y)
        self.canvas.draw_idle()


# Create the main Tkinter window
if __name__ == "__main__":
    root = tk.Tk()
    root.title("Interactive Plot in Tkinter")
    root.protocol("WM_DELETE_WINDOW", root.quit)
    app = GraphCanvas(root)
    app.pack(fill=tk.BOTH, expand=True)
    root.mainloop()
