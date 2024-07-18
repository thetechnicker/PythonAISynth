from scr.utils import DIE
from scr.predefined_functions import predefined_functions_dict
from scr.graph_canvas import GraphCanvas
from scr.fourier_neural_network_gui import NeuralNetworkGUI
from scr.fourier_neural_network import FourierNN
from _version import version
import atexit
from multiprocessing import Manager, Process, Queue
import multiprocessing
import os
from threading import Thread
import tkinter as tk
from tkinter import ttk
import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
from tkinter import messagebox
import psutil
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}


class MainGUI(tk.Tk):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.process_monitor = psutil.Process()
        self.title(f"AI Synth - {version}")
        self.rowconfigure(1, weight=1)
        self.columnconfigure(0, weight=1)
        self.columnconfigure(1, weight=1)
        self.columnconfigure(2, weight=1)
        self.columnconfigure(3, weight=1)

        self.trainings_process: Process = None
        self.music_process: Process = None
        self.training_started = False
        self.queue = Queue(-1)
        self.fourier_nn = None

        self.create_menu()
        self.create_row_one()
        self.graph = GraphCanvas(self, (900, 300))  # , draw_callback)
        self.graph.grid(row=1, column=0, columnspan=3, sticky='NSEW')
        self.add_net_controll()
        self.create_status_bar()

    def add_net_controll(self):
        defaults = {
            'SAMPLES': 44100//2,
            'EPOCHS': 100,
            'DEFAULT_FORIER_DEGREE': 250,
            'FORIER_DEGREE_DIVIDER': 1,
            'FORIER_DEGREE_OFFSET': 0,
            'PATIENCE': 10,
            'OPTIMIZER': 'Adam',
            'LOSS_FUNCTION': 'Huber',
        }

        self.gui = NeuralNetworkGUI(self, defaults=defaults)
        self.gui.grid(row=1, column=3, rowspan=3, sticky='NSEW')

    def create_status_bar(self):
        # Create a status bar with two labels
        self.status_bar = tk.Frame(self, bd=1, relief=tk.SUNKEN)
        # Adjust row and column as needed
        self.status_bar.grid(row=2, column=0, sticky='we', columnspan=4)

        self.status_label = tk.Label(
            self.status_bar, text="Ready", anchor=tk.W)
        self.status_label.pack(side=tk.LEFT)

        self.processes_label = tk.Label(
            self.status_bar, text="Processes: 0", anchor=tk.E)
        self.processes_label.pack(side=tk.RIGHT)

        self.after(500, self.update_status_bar)

    def update_status_bar(self):
        children = self.process_monitor.children(recursive=True)
        self.status_label.config(
            text="Ready" if len(children) == 0 else "Busy")
        self.processes_label.config(text=f"Processes: {len(children)}")
        self.after(500, self.update_status_bar)

    def create_row_one(self):
        self.label = tk.Label(self, text="Predefined Functions:")
        self.label.grid(row=0, column=0, sticky='NSEW')

        self.combo_selected_value = tk.StringVar()
        # Create a select box
        self.combo = ttk.Combobox(
            self, values=list(predefined_functions_dict.keys()),
            textvariable=self.combo_selected_value, state="readonly")

        self.combo.bind(
            '<<ComboboxSelected>>', self.draw_graph_from_func)
        self.combo.grid(row=0, column=1, sticky='NSEW')

        # Create a 'Start Training' button
        self.train_button = tk.Button(
            self, text="Start Training", command=self.start_training)
        self.train_button.grid(row=0, column=2, sticky='NSEW')

        # Create a 'Play' button
        self.play_button = tk.Button(
            self, text="Play", command=self.play_music)
        self.play_button.grid(row=0, column=3, sticky='NSEW')

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

    def train_update(self):
        # print("!!!check!!!")
        if self.trainings_process and self.training_started:
            if self.trainings_process.is_alive():
                try:
                    data = self.queue.get_nowait()
                    print("!!!check!!!")
                    data = list(zip(self.graph.lst, data.reshape(-1)))
                    # graph.data=data
                    self.graph.draw_extern_graph_from_data(
                        data, "training", color="red", width=self.graph.point_radius/4)
                except:
                    pass
            else:
                exit_code = self.trainings_process.exitcode
                if exit_code == None:
                    self.after(100, self.train_update)
                    return
                if exit_code == 0:
                    print("loading trained model")
                    self.fourier_nn.load_tmp_model()
                    print("model loaded")
                    self.graph.draw_extern_graph_from_func(
                        self.fourier_nn.predict, "training", color="red", width=self.graph.point_radius/4)  # , graph_type='crazy'
                DIE(self.trainings_process)
                self.trainings_process = None
                self.training_started = False
                messagebox.showinfo(
                    "training Ended", f"exit code: {exit_code}")
                return
        self.after(100, self.train_update)

    def init_or_update_nn(self):
        if not self.fourier_nn:
            self.fourier_nn = FourierNN(self.graph.export_data())
        else:
            self.fourier_nn.update_data(self.graph.export_data())
        self.fourier_nn.save_tmp_model()
        self.trainings_process = multiprocessing.Process(
            target=self.fourier_nn.train, args=(self.graph.lst, self.queue,))
        self.trainings_process.start()

    # Define your functions here

    def start_training(self):
        if self.trainings_process or self.training_started:
            print('already training')
            return
        self.training_started = True
        print("STARTING TRAINING")
        t = Thread(target=self.init_or_update_nn)
        t.daemon = True
        t.start()
        self.train_update()
        atexit.register(DIE, self.trainings_process)
        self.graph.draw_extern_graph_from_data(
            self.graph.export_data(), "train_data", color="blue")

    def play_music(self):
        print("play_music")

    def clear_graph(self):
        print("clear_graph")
        self.graph.clear()

    def export_neural_net(self):
        print("export_neural_net")

    def load_neural_net(self):
        print("load_neural_net")

    def redraw_graph(self):
        print("redraw_graph")

    def draw_graph_from_func(self, *args, **kwargs):
        print("draw_graph_from_func:", self.combo_selected_value.get())
        self.graph.use_preconfig_drawing(
            predefined_functions_dict[self.combo_selected_value.get()])


if __name__ == "__main__":
    window = MainGUI()
    window.mainloop()
    # manager.
