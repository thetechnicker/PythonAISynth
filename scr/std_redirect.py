from copy import copy
from multiprocessing import Queue
import sys
import tkinter as tk
from tkinter import scrolledtext
import queue
import time


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
        self.queue = Queue(-1)
        self.old_stdout = copy(sys.stdout.write)
        self.old_stderr = copy(sys.stderr.write)
        sys.stdout.write = self.redirector
        # sys.stderr.write = self.redirector_err
        self.after(100, self.check_queue)

    def redirector(self, inputStr):
        self.textbox.configure(state='normal')
        self.textbox.insert(tk.INSERT, inputStr)
        self.textbox.configure(state='disabled')
        self.textbox.see(tk.END)  # Auto-scroll to the end
        self.old_stdout(inputStr)

    def on_resize(self, event):
        width = self.textbox.winfo_width()
        font_width = self.textbox.tk.call(
            "font", "measure", self.textbox["font"], "-displayof", self.textbox, "0")
        sys.stdout.write = self.redirector

    def check_queue(self):
        while True:
            try:
                msg = self.queue.get_nowait()
                self.redirector(msg)
            except queue.Empty:
                break
        self.after(100, self.check_queue)
