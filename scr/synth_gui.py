import mido
import tkinter as tk
from tkinter import ttk


class SynthGUI(ttk.Frame):
    def __init__(self, parent=None, **kwargs):
        ttk.Frame.__init__(self, parent, **kwargs)
        self.columnconfigure(0, weight=1)
        self.columnconfigure(1, weight=1)


        param="MIDI PORT"
        label = ttk.Label(self, text=param)
        label.grid(row=0, column=0, sticky='NSW')
        options=mido.get_input_names()
        self.midi_port_var=tk.StringVar(self, options[0])
        self.midi_port_option_menu = ttk.OptionMenu(
                    self, self.midi_port_var,  options[0], *(options))
        self.midi_port_option_menu.grid(row=0, column=1, sticky='NSE')
        # self.midi_port_var.trace_add('write', lambda *args, key=param,
        #                 var=self.midi_port_var: self.on_change(key, var.get(), var))

    def get_port_name(self):
        return self.midi_port_var.get()
    
    
    # def on_change(self, name, value, var):
    #     if hasattr(self, 'on_change_callback'):
    #         if self.on_change_callback:
    #             worked = self.on_change_callback(name, value)
    #             # print(worked)
    #             if worked:
    #                 self.previous_values[name] = value
    #             else:
    #                 var.set(self.previous_values[name])

    # def set_on_change(self, func):
    #     self.on_change_callback = func
    #     self.previous_values = {param: var.get()
    #                             for param, var in self.params.items()}

