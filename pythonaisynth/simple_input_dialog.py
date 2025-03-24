import tkinter as tk
from tkinter import simpledialog, ttk


class EntryWithPlaceholder(tk.Entry):
    def __init__(self, master=None, placeholder="PLACEHOLDER", color="grey"):
        super().__init__(master)
        self.placeholder = placeholder
        self.placeholder_color = color
        self.default_fg_color = self["fg"]

        self.bind("<FocusIn>", self.foc_in)
        self.bind("<FocusOut>", self.foc_out)

        self.put_placeholder()

    def put_placeholder(self):
        self.insert(0, self.placeholder)
        self["fg"] = self.placeholder_color

    def foc_in(self, *args):
        if self["fg"] == self.placeholder_color:
            self.delete("0", "end")
            self["fg"] = self.default_fg_color

    def foc_out(self, *args):
        if not self.get():
            self.put_placeholder()


class askStringAndSelectionDialog(simpledialog.Dialog):
    def __init__(
        self,
        parent,
        title=None,
        label_str="",
        default_str="",
        label_select="",
        default_select="",
        values_to_select_from=[],
    ):
        self.label_str = label_str
        self.default_str = default_str
        self.label_select = label_select
        self.default_select = default_select
        self.values_to_select_from = values_to_select_from
        super().__init__(parent, title=title)

    def body(self, master):
        tk.Label(master, text=self.label_str).grid(row=0)
        self.entry = EntryWithPlaceholder(master, placeholder=self.default_str)
        self.entry.grid(row=0, column=1)

        tk.Label(master, text=self.label_select).grid(row=1)
        self.combo = ttk.Combobox(master, values=self.values_to_select_from)
        self.combo.set(self.default_select)
        self.combo.grid(row=1, column=1)

        return self.entry  # initial focus

    def apply(self):
        self.result = (self.entry.get(), self.combo.get())
