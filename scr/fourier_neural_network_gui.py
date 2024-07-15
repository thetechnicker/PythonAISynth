import tkinter as tk
from tensorflow.keras import losses, optimizers


class NeuralNetworkGUI(tk.Frame):
    def __init__(self, parent=None, defaults: dict = None, callback=None, **kwargs):
        tk.Frame.__init__(self, parent, **kwargs)
        # self.grid(sticky='NSEW')
        self.on_change_callback=callback
        self.columnconfigure(0, weight=1)
        self.columnconfigure(1, weight=1)

        # Define the list of loss functions and optimizers
        self.loss_functions = [func for func in dir(
            losses) if not func.startswith('_') and isinstance(getattr(losses, func), type)]

        self.optimizers_list = [opt for opt in dir(
            optimizers) if not opt.startswith('_') and isinstance(getattr(optimizers, opt), type)]
        
        # print(self.loss_functions, self.optimizers_list, sep='\n-------------------------------\n')

        # Create labels and entry fields for the parameters
        self.params = ['SAMPLES', 'EPOCHS', 'DEFAULT_FORIER_DEGREE', 'CALC_FOURIER_DEGREE_BY_DATA_LENGTH',
                       'FORIER_DEGREE_DIVIDER', 'FORIER_DEGREE_OFFSET', 'PATIENCE', 'OPTIMIZER', 'LOSS_FUNCTION']
        for i, param in enumerate(self.params):
            # self.rowconfigure(i, weight=1)
            label = tk.Label(self, text=param)
            label.grid(row=i, column=0, sticky='NSEW')
            default = defaults.get(
                param, 0 if param != 'CALC_FOURIER_DEGREE_BY_DATA_LENGTH' else False)
            var = tk.BooleanVar(self, value=default) if param == 'CALC_FOURIER_DEGREE_BY_DATA_LENGTH'  else tk.StringVar(self, value=default) if param in ['OPTIMIZER', 'LOSS_FUNCTION'] else tk.IntVar(
                self, value=default)
            var.trace_add('write', lambda *args, key=param,
                          var=var: self.on_change(key, var.get()))
            if param in ['OPTIMIZER', 'LOSS_FUNCTION']:
                # set the default option
                used_list = self.optimizers_list if param == 'OPTIMIZER' else self.loss_functions
                try:
                    default = used_list[used_list.index(defaults.get(param))]
                except ValueError:
                    default = used_list[0]

                var.set(default)

                dropdown = tk.OptionMenu(
                    self, var, *(used_list))
                dropdown.grid(row=i, column=1, sticky='NSEW')
            else:
                entry = tk.Entry(self, textvariable=var)
                entry.grid(row=i, column=1, sticky='NSEW')

    def on_change(self, name, value):
        # print(f"{name} changed to {value}")
        if hasattr(self, 'on_change_callback'):
            if self.on_change_callback:
                self.on_change_callback(name, value)

    def set_on_change(self, func):
        self.on_change_callback=func


if __name__ == "__main__":
    # Usage
    root = tk.Tk()
    root.rowconfigure(0, weight=1)
    root.columnconfigure(0, weight=1)
    defaults = {
        'SAMPLES': 44100//2,
        'EPOCHS': 100,
        'DEFAULT_FORIER_DEGREE': 250,
        'FORIER_DEGREE_DIVIDER': 1,
        'FORIER_DEGREE_OFFSET': 0,
        'PATIENCE': 10,
        'OPTIMIZER': 'Adam',
        'LOSS_FUNCTION': 'Huber'
    }
    gui = NeuralNetworkGUI(root, defaults=defaults)
    gui.grid(row=0, column=0, sticky='NSEW')
    root.mainloop()
