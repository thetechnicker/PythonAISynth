import torch.optim as optim
import torch.nn as nn
import tkinter as tk
from tkinter import ttk


class NeuralNetworkGUI(ttk.Frame):
    def __init__(self, parent=None, defaults: dict = None, callback=None, **kwargs):
        ttk.Frame.__init__(self, parent, **kwargs)
        self.on_change_callback = callback
        self.columnconfigure(0, weight=1)
        self.columnconfigure(1, weight=1)

        # Define the list of loss functions and optimizers
        self.loss_functions = [
            func
            for func in dir(nn)
            if not func.startswith("_")
            and isinstance(getattr(nn, func), type)
            and issubclass(getattr(nn, func), nn.modules.loss._Loss)
        ]
        self.loss_functions.append("CustomHuberLoss")
        self.optimizers_list = [
            opt
            for opt in dir(optim)
            if not opt.startswith("_")
            and isinstance(getattr(optim, opt), type)
            and issubclass(getattr(optim, opt), optim.Optimizer)
        ]
        print(
            "".center(40, "-"),
            *self.loss_functions,
            "".center(40, "-"),
            *self.optimizers_list,
            "".center(40, "-"),
            sep="\n"
        )
        # exit(0)
        # Create labels and entry fields for the parameters
        self.params = [
            "SAMPLES",
            "EPOCHS",
            "DEFAULT_FORIER_DEGREE",
            "CALC_FOURIER_DEGREE_BY_DATA_LENGTH",
            "FORIER_DEGREE_DIVIDER",
            "FORIER_DEGREE_OFFSET",
            "PATIENCE",
            "OPTIMIZER",
            "LOSS_FUNCTION",
        ]

        self.previous_values = {}

        for i, param in enumerate(self.params):
            label = ttk.Label(self, text=param)
            label.grid(row=i, column=0, sticky="NSW")
            default = defaults.get(
                param, 0 if param != "CALC_FOURIER_DEGREE_BY_DATA_LENGTH" else False
            )
            var = (
                tk.BooleanVar(self, value=default)
                if param == "CALC_FOURIER_DEGREE_BY_DATA_LENGTH"
                else (
                    tk.StringVar(self, value=default)
                    if param in ["OPTIMIZER", "LOSS_FUNCTION"]
                    else tk.IntVar(self, value=default)
                )
            )
            var.trace_add(
                "write",
                lambda *args, key=param, var=var: self.on_change(key, var.get(), var),
            )
            self.previous_values[param] = default
            if param in ["OPTIMIZER", "LOSS_FUNCTION"]:
                # set the default option
                used_list = (
                    self.optimizers_list
                    if param == "OPTIMIZER"
                    else self.loss_functions
                )
                try:
                    default = used_list[used_list.index(defaults.get(param))]
                except ValueError:
                    default = used_list[0]

                var.set(default)

                dropdown = ttk.OptionMenu(self, var, *(used_list))
                dropdown.grid(row=i, column=1, sticky="NSEW")
            elif param == "CALC_FOURIER_DEGREE_BY_DATA_LENGTH":
                entry = ttk.Checkbutton(self, variable=var)
                entry.grid(row=i, column=1, sticky="NSEW")
            else:
                entry = ttk.Entry(self, textvariable=var)
                entry.grid(row=i, column=1, sticky="NSEW")
            var.set(default)

    def on_change(self, name, value, var):
        if hasattr(self, "on_change_callback"):
            if self.on_change_callback:
                worked = self.on_change_callback(name, value)
                # print(worked)
                if worked:
                    self.previous_values[name] = value
                else:
                    var.set(self.previous_values[name])

    def set_on_change(self, func):
        self.on_change_callback = func
        self.previous_values = {param: var.get() for param, var in self.params.items()}


if __name__ == "__main__":

    def stupid(*args, **kwargs):
        print(*args, *kwargs.items(), sep="\n")
        return False

    # Usage
    root = tk.Tk()
    root.rowconfigure(0, weight=1)
    root.columnconfigure(0, weight=1)
    defaults = {
        "SAMPLES": 44100 // 2,
        "EPOCHS": 100,
        "DEFAULT_FORIER_DEGREE": 250,
        "FORIER_DEGREE_DIVIDER": 1,
        "FORIER_DEGREE_OFFSET": 0,
        "PATIENCE": 10,
        "OPTIMIZER": "Adam",
        "LOSS_FUNCTION": "mse_loss",
    }
    gui = NeuralNetworkGUI(root, defaults=defaults, callback=stupid)
    gui.grid(row=0, column=0, sticky="NSEW")
    root.mainloop()

    exit()

    # autopep8: off

    loss_functions = [
        func
        for func in dir(nn)
        if not func.startswith("_")
        and isinstance(getattr(nn, func), type)
        and issubclass(getattr(nn, func), nn.modules.loss._Loss)
    ]
    print(loss_functions)

    optimizers_list = [
        opt
        for opt in dir(optim)
        if not opt.startswith("_")
        and isinstance(getattr(optim, opt), type)
        and issubclass(getattr(optim, opt), optim.Optimizer)
    ]
    print(optimizers_list)
