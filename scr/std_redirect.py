import sys
import tkinter as tk
from tkinter import scrolledtext


class RedirectedOutputFrame(tk.Frame):
    def __init__(self, master=None):
        super().__init__(master)
        self.textbox = scrolledtext.ScrolledText(self, height=10)
        self.textbox.pack(fill=tk.BOTH, expand=True)
        sys.stdout.write = self.redirector_stdout
        sys.stderr.write = self.redirector_stderr

    def redirector_stdout(self, inputStr):
        self.textbox.insert(tk.INSERT, inputStr)

    def redirector_stderr(self, inputStr):
        self.textbox.insert(tk.INSERT, inputStr, 'error')
        self.textbox.tag_config('error', foreground='red')
