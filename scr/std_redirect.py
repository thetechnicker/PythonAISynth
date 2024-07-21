import sys
import tkinter as tk
from tkinter import scrolledtext


class RedirectedOutputFrame(tk.Frame):
    def __init__(self, master=None):
        super().__init__(master)
        self.textbox = scrolledtext.ScrolledText(self, height=10)
        self.textbox.pack(fill=tk.BOTH, expand=True)
        # Make the textbox non-editable
        self.textbox.configure(state='disabled')
        # Wrap text at the end of the window
        self.textbox.configure(wrap='word')
        # Bind the resize event
        self.textbox.bind("<Configure>", self.on_resize)
        sys.stdout.write = self.redirector_stdout
        sys.stderr.write = self.redirector_stderr

    def redirector_stdout(self, inputStr):
        self.textbox.configure(state='normal')
        self.textbox.insert(tk.INSERT, inputStr)
        self.textbox.configure(state='disabled')
        self.textbox.see(tk.END)  # Auto-scroll to the end

    def redirector_stderr(self, inputStr):
        self.textbox.configure(state='normal')
        self.textbox.insert(tk.INSERT, inputStr, 'error')
        self.textbox.configure(state='disabled')
        self.textbox.see(tk.END)  # Auto-scroll to the end
        self.textbox.tag_config('error', foreground='red')

    def on_resize(self, event):
        width = self.textbox.winfo_width()
        font_width = self.textbox.tk.call(
            "font", "measure", self.textbox["font"], "-displayof", self.textbox, "0")
        sys.stdout.write = self.redirector_stdout
