import tkinter as tk
from tkinter import ttk
import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
from tkinter import messagebox

from _version import version
from scr.predefined_functions import predefined_functions_dict
# from scr.graph_canvas import GraphCanvas
# from scr.simple_input_dialog import askStringAndSelectionDialog
# from scr.fourier_neural_network_gui import NeuralNetworkGUI


class MainGUI(tk.Tk):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.title(f"AI Synth - {version}")
        self.rowconfigure(1, weight=1)
        self.create_menu()
        self.create_row_one()

    def create_row_one(self):
        self.label = tk.Label(self, text="Predefined Functions:")
        self.label.grid(row=0, column=0)

        self.combo_selected_value = tk.StringVar()
        # Create a select box
        self.combo = ttk.Combobox(
            self, values=list(predefined_functions_dict.keys()),
            textvariable=self.combo_selected_value, state="readonly")

        self.combo.bind(
            '<<ComboboxSelected>>', self.draw_graph_from_func)
        self.combo.grid(row=0, column=1)

        # Create a 'Start Training' button
        self.train_button = tk.Button(
            self, text="Start Training", command=self.start_training)
        self.train_button.grid(row=0, column=2)

        # Create a 'Play' button
        self.play_button = tk.Button(
            self, text="Play", command=self.play_music)
        self.play_button.grid(row=0, column=3)

    def create_menu(self):
        # Create a menu bar
        self.menu_bar = tk.Menu(self)

        # Create a 'File' menu
        self.create_menu_item("File", [
            ("Export Neural Net", self.export_neural_net),
            ("Load Neural Net", self.load_neural_net)
        ])

        # Create a 'Training' menu
        self.create_menu_item("Training", [
            ("Start Training", self.start_training)
        ])

        # Create a 'Graph' menu
        self.create_menu_item("Graph", [
            ("Clear Graph", self.clear_graph),
            ("Redraw Graph", self.redraw_graph)
        ])

        # Create a 'Music' menu
        self.create_menu_item("Music", [
            ("Play Music from MIDI", self.play_music)
        ])

        # Display the menu
        self.config(menu=self.menu_bar)

    def create_menu_item(self, name, commands):
        menu = tk.Menu(self.menu_bar, tearoff=0)
        for label, command in commands:
            menu.add_command(label=label, command=command)
        self.menu_bar.add_cascade(label=name, menu=menu)

    # Define your functions here
    def start_training(self):
        print("start_training")

    def play_music(self):
        print("play_music")

    def clear_graph(self):
        print("clear_graph")

    def export_neural_net(self):
        print("export_neural_net")

    def load_neural_net(self):
        print("load_neural_net")

    def redraw_graph(self):
        print("redraw_graph")

    def draw_graph_from_func(self, *args, **kwargs):
        print("draw_graph_from_func:", self.combo_selected_value.get())


if __name__ == "__main__":
    window = MainGUI()
    window.mainloop()
