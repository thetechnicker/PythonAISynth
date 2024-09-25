import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.widgets import Cursor
import tkinter as tk
from tkinter import ttk

from scr.utils import run_after_ms


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
        self.ax.set_ylim(-1, 1)

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

        # Connect the events to the functions
        self.fig.canvas.mpl_connect('button_press_event', self.on_mouse_press)
        self.fig.canvas.mpl_connect(
            'button_release_event', self.on_mouse_release)
        self.fig.canvas.mpl_connect(
            'motion_notify_event', self.on_mouse_motion)

        # Create a canvas to embed the plot in Tkinter

        self.redraw_needed = False
        run_after_ms(100, self.draw_idle)

        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def draw_idle(self):
        if self.redraw_needed:
            self.canvas.draw_idle()
            run_after_ms(10, self.draw_idle)
        else:
            run_after_ms(100, self.draw_idle)

    # Function to find the closest point to the mouse

    def find_closest_point(self, x_mouse):
        distances = np.sqrt((self.x - x_mouse) ** 2)
        return np.argmin(distances)

    # Function to update the y-values of the selected dot
    def update_y(self, event):
        if event.inaxes == self.ax and self.selected_index is not None:
            # Get the y-coordinate of the mouse event
            y_new = event.ydata
            # Update the y-value of the selected dot
            self.y[self.selected_index] = y_new
            self.line.set_ydata(self.y)
            self.canvas.draw_idle()

    # Function to handle mouse button press
    def on_mouse_press(self, event):
        if event.inaxes == self.ax and event.button == 1:  # Left mouse button
            self.selected_index = self.find_closest_point(event.xdata)
            self.last_index = self.selected_index  # Store the last index
            # Jump to mouse y position
            self.y[self.selected_index] = event.ydata
            self.line.set_ydata(self.y)
            # self.canvas.draw_idle()
            self.redraw_needed = True

    # Function to handle mouse motion
    def on_mouse_motion(self, event):
        if event.inaxes == self.ax and self.selected_index is not None:
            # Find the closest point again based on the current mouse position
            new_index = self.find_closest_point(event.xdata)
            if new_index != self.selected_index:
                # Interpolate between last_index and new_index
                if self.last_index is not None:
                    start_index = self.last_index
                    end_index = new_index
                    indices = np.arange(
                        start_index, end_index + 1) if new_index > self.selected_index else np.arange(start_index, end_index - 1, -1)
                    percentages = (indices - start_index) / \
                        (end_index - start_index)
                    self.y[indices] = event.ydata * percentages + \
                        self.y[start_index] * (1 - percentages)
                    # start_index = self.last_index
                    # end_index = new_index
                    # if new_index > self.selected_index:
                    #     for i in range(start_index, end_index + 1):
                    #         percentage = (i - start_index) / \
                    #             (end_index - start_index)
                    #         # Linear interpolation
                    #         self.y[i] = event.ydata * percentage + \
                    #             self.y[start_index] * (1 - percentage)
                    # else:
                    #     for i in range(start_index, end_index - 1, -1):
                    #         percentage = (i - start_index) / \
                    #             (end_index - start_index)
                    #         # Linear interpolation
                    #         self.y[i] = event.ydata * percentage + \
                    #             self.y[start_index] * (1 - percentage)
                self.selected_index = new_index
                self.line.set_ydata(self.y)
                self.last_index = new_index
            self.redraw_needed = True
            # self.canvas.draw_idle()

    # Function to handle mouse button release
    def on_mouse_release(self, event):
        self.last_index = self.selected_index  # Update last index on release
        self.selected_index = None  # Reset the selected index
        self.redraw_needed = False

    def destroy(self):
        plt.close(self.fig)  # Close the matplotlib figure
        super().destroy()  # Call the parent destroy method


# Create the main Tkinter window
if __name__ == "__main__":
    root = tk.Tk()
    root.title("Interactive Plot in Tkinter")
    # root.protocol("WM_DELETE_WINDOW", root.quit)
    app = GraphCanvas(root)
    app.pack(fill=tk.BOTH, expand=True)
    root.mainloop()
