import copy
import socket
import sys
import threading
import time

import numpy as np
from scr import music
from scr import utils
from scr.music import Synth2, musik_from_file
from scr.simple_input_dialog import askStringAndSelectionDialog
from scr.std_redirect import RedirectedOutputFrame
from scr.utils import DIE, tk_after_errorless
from scr.predefined_functions import predefined_functions_dict
from scr.graph_canvas_v2 import GraphCanvas
from scr.fourier_neural_network_gui import NeuralNetworkGUI
from scr.fourier_neural_network import FourierNN
from _version import version
import atexit
from multiprocessing import Process, Queue
from multiprocessing.managers import SyncManager
import multiprocessing
import os
from threading import Thread
import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
from tkinter import messagebox
import psutil

state = "bootup"
window = None

dark_theme = {
    ".": {
        "configure": {
            "background": "#2d2d2d",  # Dark grey background
            "foreground": "white",    # White text
        }
    },
    "TLabel": {
        "configure": {
            "foreground": "white",    # White text
        }
    },
    "TButton": {
        "configure": {
            "background": "#3c3f41",  # Dark blue-grey button
            "foreground": "white",    # White text
        }
    },
    "TEntry": {
        "configure": {
            "background": "#2d2d2d",  # Dark grey background
            "foreground": "white",    # White text
            "fieldbackground": "#4d4d4d",
            "insertcolor": "white",
            "bordercolor": "black",
            "lightcolor": "#4d4d4d",
            "darkcolor": "black",
        }
    },
    "TCheckbutton": {
        "configure": {
            "foreground": "white",    # White text
            "indicatorbackground": "white",
            "indicatorforeground": "black",
        }
    },
    "TCombobox": {
        "configure": {
            "background": "#2d2d2d",  # Dark grey background
            "foreground": "white",    # White text
            "fieldbackground": "#4d4d4d",
            "insertcolor": "white",
            "bordercolor": "black",
            "lightcolor": "#4d4d4d",
            "darkcolor": "black",
            "arrowcolor": "white"
        },
    },
}

DARKMODE = True


class MainGUI(tk.Tk):
    def __init__(self, *args, manager: SyncManager = None, **kwargs):
        super().__init__(*args, **kwargs)

        self.style = ttk.Style()
        if DARKMODE:
            self.configure(bg='#202020')
            self.option_add("*TCombobox*Listbox*Background", "#202020")
            self.option_add("*TCombobox*Listbox*Foreground", "white")
            self.style.theme_create('dark', parent="clam", settings=dark_theme)
            self.style.theme_use('dark')

        self.manager = manager
        self.lock = manager.Lock()
        self.std_queue = manager.Queue(-1)
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
        self.block_training = False
        self.queue = Queue(-1)
        self.fourier_nn = None
        self.synth: Synth2 = None
        self.init_terminal_frame()
        self.create_menu()
        self.create_row_one()
        self.graph = GraphCanvas(self, (900, 300), DARKMODE)
        self.graph.grid(row=1, column=0, columnspan=3, sticky='NSEW')
        self.create_controll_column()
        self.create_status_bar()
        # sys.stdout = utils.QueueSTD_OUT(self.std_queue)

    # def destroy(self):
    #     self.graph.destroy()
    #     print("CLOSING")
    #     time.sleep(1+max(tk_after_errorless.max_time)/1000)

    #     super().destroy()

    def init_terminal_frame(self):
        self.std_redirect = RedirectedOutputFrame(self, self.std_queue)
        # self.std_queue = self.std_redirect.queue
        self.std_redirect.grid(row=2, column=0, columnspan=3, sticky='NSEW')

    def create_controll_column(self):
        self.frame = ttk.Frame(self)

        defaults = {
            'SAMPLES': 500,
            'EPOCHS': 1000,
            'DEFAULT_FORIER_DEGREE': 100,
            'FORIER_DEGREE_DIVIDER': 1,
            'FORIER_DEGREE_OFFSET': 0,
            'PATIENCE': 50,
            'OPTIMIZER': 'SGD',
            'LOSS_FUNCTION': 'HuberLoss',
        }

        self.gui = NeuralNetworkGUI(
            self.frame, defaults=defaults, callback=self.update_frourier_params)
        self.gui.grid(row=0, sticky='NSEW')
        self.frame.grid(row=1, column=3, rowspan=2, sticky='NSEW')

    def create_status_bar(self):
        # Create a status bar with two labels
        self.status_bar = ttk.Frame(self, relief=tk.SUNKEN)
        # Adjust row and column as needed
        self.status_bar.grid(row=3, column=0, sticky='we', columnspan=4)

        self.status_label = ttk.Label(
            self.status_bar, text="Ready", anchor=tk.W, font=("TkFixedFont"), relief=tk.SUNKEN)
        self.status_label.pack(side=tk.LEFT)

        self.processes_label = ttk.Label(
            self.status_bar, text="Children Processes: 0", anchor=tk.E, font=("TkFixedFont"), relief=tk.SUNKEN)
        self.processes_label.pack(side=tk.RIGHT)

        # CPU and RAM
        self.cpu_label = ttk.Label(
            self.status_bar, text="CPU Usage: 0%", anchor=tk.E, font=("TkFixedFont"), relief=tk.SUNKEN)
        self.cpu_label.pack(side=tk.RIGHT)

        self.ram_label = ttk.Label(
            self.status_bar, text="RAM Usage: 0%", anchor=tk.E, font=("TkFixedFont"), relief=tk.SUNKEN)
        self.ram_label.pack(side=tk.RIGHT)

        self.frame_no = 0
        tk_after_errorless(self, 500, self.update_status_bar)

    def update_status_bar(self):
        children = self.process_monitor.children(recursive=True)
        total_cpu = self.process_monitor.cpu_percent()
        total_mem = self.process_monitor.memory_percent()
        i = 0
        for child in children:
            try:
                total_cpu += child.cpu_percent(0.01)
                total_mem += child.memory_percent()
                i += 1
            except psutil.NoSuchProcess:
                pass
        # total_cpu /= i
        # total_mem /= i
        if total_mem >= 90:
            print(total_mem)
            self.quit()
            # input("press key to continue")
        self.cpu_label.config(
            text=f"CPU Usage: {total_cpu/os.cpu_count():.2f}%")
        self.ram_label.config(text=f"RAM Usage: {total_mem:.2f}%")

        animation_text = "|" if self.frame_no == 0 else '/' if self.frame_no == 1 else '-' if self.frame_no == 2 else '\\'
        self.frame_no = (self.frame_no+1) % 4
        if len(children) <= 1:
            self.status_label.config(text="Ready", foreground="green")
        else:
            self.status_label.config(
                text=f"Busy ({animation_text})", foreground="red")

        self.processes_label.config(
            text=f"Children Processes: {len(children)}")
        tk_after_errorless(self, 500, self.update_status_bar)

    def create_row_one(self):
        self.label = ttk.Label(self, text="Predefined Functions:")
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
        self.train_button = ttk.Button(
            self, text="Start Training", command=self.start_training)
        self.train_button.grid(row=0, column=2, sticky='NSEW')

        # Create a 'Play' button
        self.play_button = ttk.Button(
            self, text="Play", command=self.play_music)
        self.play_button.grid(row=0, column=3, sticky='NSEW')

    def create_menu(self):
        # Create a menu bar
        if DARKMODE:
            self.menu_bar = tk.Menu(self, bg="#2d2d2d", fg="white",
                                    activebackground="#3e3e3e", activeforeground="white")
        else:
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
            # ("Redraw Graph", self.redraw_graph)
        ])

        # Create a 'Music' menu
        self.create_menu_item("Music", [
            ("Play Music from MIDI Port", self.play_music),
            ("Play Music from MIDI File", self.play_music_file)
        ])

        # Display the menu
        self.config(menu=self.menu_bar)

    def create_menu_item(self, name, commands):
        if DARKMODE:
            menu = tk.Menu(self.menu_bar, tearoff=0, bg="#2d2d2d", fg="white",
                           activebackground="#3e3e3e", activeforeground="white")
        else:
            menu = tk.Menu(self.menu_bar, tearoff=0)
        for label, command in commands:
            menu.add_command(label=label, command=command)
        self.menu_bar.add_cascade(label=name, menu=menu)

    def train_update(self):
        # print("!!!check!!!")
        if self.trainings_process and self.training_started:
            if self.trainings_process.is_alive() or not self.queue.empty():
                try:
                    data = self.queue.get_nowait()
                    # print("!!!check!!!")
                    data = list(zip(self.graph.x, data))
                    self.graph.plot_points(data, name="training", type='line')
                except Exception as e:
                    # print(e)
                    pass
            else:
                exit_code = self.trainings_process.exitcode
                if exit_code == None:
                    tk_after_errorless(self, 100, self.train_update)
                    return
                if exit_code == 0:
                    print("loading trained model")
                    self.fourier_nn.load_new_model_from_file(delete_tmp=True)
                    print("model loaded")
                    # for name, param in self.fourier_nn.current_model.named_parameters():
                    #     print(
                    #         f"Layer: {name} | Size: {param.size()} | Values:\n{param[:2]}\n------------------------------")
                    self.graph.plot_function(
                        self.fourier_nn.predict, x_range=(0, 2*np.pi, 44100//2))
                    self.synth = Synth2(self.fourier_nn, self.std_queue)
                DIE(self.trainings_process)
                self.trainings_process = None
                self.training_started = False
                self.block_training = False
                messagebox.showinfo(
                    "training Ended", f"exit code: {exit_code}")
                # self.fourier_nn.clean_memory()
                return
        tk_after_errorless(self, 10, self.train_update)

    def init_or_update_nn(self, stdout=None):
        if not self.fourier_nn:
            print("".center(20, '*'))
            self.fourier_nn = FourierNN(
                self.lock, self.graph.export_data(), stdout_queue=stdout)
        else:
            print("".center(20, '#'))
            self.fourier_nn.update_data(self.graph.export_data())

        self.fourier_nn.save_model()
        self.trainings_process = multiprocessing.Process(
            target=self.fourier_nn.train, args=(self.graph.x, self.queue, ), kwargs={"stdout_queue": self.std_queue})
        self.trainings_process.start()
        self.training_started = True

    # Define your functions here

    def start_training(self):
        if self.trainings_process or self.block_training:
            print('already training')
            return
        if self.synth:
            if self.synth.live_synth:
                self.synth.run_live_synth()
        self.block_training = True
        print("STARTING TRAINING")
        t = Thread(target=self.init_or_update_nn, args=(self.std_queue,))
        t.daemon = True
        t.start()
        tk_after_errorless(self, 100, self.train_update)
        atexit.register(DIE, self.trainings_process)
        self.graph.remove_plot("predict")

    def play_music(self):
        print("play_music")
        if self.synth:
            self.synth.run_live_synth()
        else:
            print("or not")
            # music.midi_to_musik_live(self.fourier_nn, self.std_queue)

    def play_music_file(self):
        print("play_music")
        if self.fourier_nn:
            music.musik_from_file(self.fourier_nn)
            # music.midi_to_musik_live(self.fourier_nn, self.std_queue)

    def clear_graph(self):
        print("clear_graph")
        self.graph.clear()

    def export_neural_net(self):
        print("export_neural_net")
        default_format = 'keras'
        if self.fourier_nn:
            path = './tmp'
            if not os.path.exists(path):
                os.mkdir(path)
            i = 0
            while os.path.exists(path+f"/model{i}.{default_format}"):
                i += 1

            dialog = askStringAndSelectionDialog(parent=self,
                                                 title="Save Model",
                                                 label_str="Enter a File Name",
                                                 default_str=f"model{i}",
                                                 label_select="Select Format",
                                                 default_select=default_format,
                                                 values_to_select_from=["pth"])
            name, file_format = dialog.result
            if not name:
                name = f"model{i}"

            file = f"{path}/{name}.{file_format}"
            print(file)
            if not os.path.exists(file):
                try:
                    self.fourier_nn.save_model(file)
                except Exception as e:
                    messagebox.showwarning(
                        "ERROR - Can't Save Model", f"{e}")
            else:
                messagebox.showwarning(
                    "File already Exists", f"The selected filename {name} already exists.")

    def load_neural_net(self):
        print("load_neural_net")
        filetypes = (('Torch Model File', '*.pth'),
                     ('All files', '*.*'))
        filename = filedialog.askopenfilename(
            title='Open a file', initialdir='.', filetypes=filetypes, parent=self)
        if os.path.exists(filename):
            if not self.fourier_nn:
                self.fourier_nn = FourierNN(
                    lock=self.lock, data=None, stdout_queue=self.std_queue)
            self.fourier_nn.load_new_model_from_file(filename)
            name = os.path.basename(filename).split('.')[0]
            self.graph.draw_extern_graph_from_func(
                self.fourier_nn.predict, name)
            print(name)
            self.synth = Synth2(self.fourier_nn, self.std_queue)
            # self.fourier_nn.update_data(
            #     data=self.graph.get_graph(name=name)[0])

    # def redraw_graph(self):
    #     # now idea when usefull
    #     print("redraw_graph")
    #     self.graph.draw_extern_graph_from_func(
    #         self.fourier_nn.predict, "training", color="red", width=self.graph.point_radius/4)  # , graph_type='crazy')
    #     # self.graph._draw()

    def draw_graph_from_func(self, *args, **kwargs):
        print("draw_graph_from_func:", self.combo_selected_value.get())
        self.graph.plot_function(
            predefined_functions_dict[self.combo_selected_value.get()], overwrite=True)

    def update_frourier_params(self, key, value):
        if not self.fourier_nn:
            # print(self.std_queue)
            self.fourier_nn = FourierNN(
                lock=self.lock, stdout_queue=self.std_queue)
        if not self.trainings_process and not self.block_training:
            self.fourier_nn.update_attribs(**{key: value})
            return True
        return False

# for "invalid command" in the tk.TK().after() function when programm gets closed
# def start_server():
#     global window
#     with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
#         while True:
#             try:
#                 s.bind(('localhost', 65432))
#                 break
#             except:
#                 pass
#         s.listen()
#         while True:
#             conn, addr = s.accept()
#             with conn:
#                 print('Connected by', addr)
#                 conn.sendall(state.encode())  # Send the current state
#                 msg = ""
#                 try:
#                     # Receive message from client
#                     msg = conn.recv(1024).decode()
#                     print("Received message:", msg)
#                 except Exception as e:
#                     print("Error receiving message:", e)
#                 if msg == "exit":
#                     # Close the Tkinter window
#                     window.after(10, window.destroy)


def main():
    # global state, window
    # threading.Thread(target=start_server, daemon=True).start()

    os.environ['HAS_RUN_INIT'] = 'True'
    multiprocessing.set_start_method("spawn")
    with multiprocessing.Manager() as manager:
        std_write = copy.copy(sys.stdout.write)
        window = MainGUI(manager=manager)
        window.protocol("WM_DELETE_WINDOW", window.quit)
        state = "running"
        try:
            # window.after(1000, lambda: window.graph.plot_function(
            #     predefined_functions_dict['nice'], overwrite=True))
            # window.after(2000, window.start_training)
            window.mainloop()
        except KeyboardInterrupt:
            print("exiting via keybord interupt")
            # window.quit()
        sys.stdout.write = std_write


if __name__ == "__main__":
    main()
