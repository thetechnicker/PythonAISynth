import queue
import tkinter as tk
from tkinter import scrolledtext
import re
from copy import copy
import sys
from multiprocessing import Queue

from scr import utils


class RedirectedOutputFrame(tk.Frame):
    def __init__(self, master=None, std_queue=None):
        super().__init__(master)
        self.textbox = scrolledtext.ScrolledText(
            self, height=10, font=("TkFixedFont"))
        self.textbox.pack(fill=tk.BOTH, expand=True)
        self.textbox.configure(state='disabled')
        self.textbox.configure(wrap='word')
        self.textbox.bind("<Configure>", self.on_resize)
        self.queue: Queue = Queue(-1) if not std_queue else std_queue
        self.old_stdout = copy(sys.stdout.write)
        self.old_stderr = copy(sys.stderr.write)
        sys.stdout.write = self.redirector
        sys.stderr.write = self.redirector
        self.after(100, self.check_queue)

        # dictionaries to replace formatting code with tags
        self.ansi_font_format = {1: 'bold',
                                 3: 'italic', 4: 'underline', 9: 'overstrike'}
        self.ansi_font_reset = {21: 'bold', 23: 'italic',
                                24: 'underline', 29: 'overstrike'}

        # tag configuration
        self.textbox.tag_configure('bold', font=('', 9, 'bold'))
        self.textbox.tag_configure('italic', font=('', 9, 'italic'))
        self.textbox.tag_configure('underline', underline=True)
        self.textbox.tag_configure('overstrike', overstrike=True)

        # dictionaries to replace color code with tags
        self.ansi_color_fg = {39: 'foreground default'}
        self.ansi_color_bg = {49: 'background default'}

        self.textbox.tag_configure(
            'foreground default', foreground=self.textbox["fg"])
        self.textbox.tag_configure(
            'background default', background=self.textbox["bg"])

        self.ansi_colors_dark = [
            'black', 'red', 'green', 'yellow', 'royal blue', 'magenta', 'cyan', 'light gray']
        self.ansi_colors_light = ['dark gray', 'tomato', 'light green',
                                  'light goldenrod', 'light blue', 'pink', 'light cyan', 'white']

        for i, (col_dark, col_light) in enumerate(zip(self.ansi_colors_dark, self.ansi_colors_light)):
            self.ansi_color_fg[30 + i] = 'foreground ' + col_dark
            self.ansi_color_fg[90 + i] = 'foreground ' + col_light
            self.ansi_color_bg[40 + i] = 'background ' + col_dark
            self.ansi_color_bg[100 + i] = 'background ' + col_light
            # tag configuration
            self.textbox.tag_configure(
                'foreground ' + col_dark, foreground=col_dark)
            self.textbox.tag_configure(
                'background ' + col_dark, background=col_dark)
            self.textbox.tag_configure(
                'foreground ' + col_light, foreground=col_light)
            self.textbox.tag_configure(
                'background ' + col_light, background=col_light)

        # regular expression to find ansi codes in string
        self.ansi_regexp = re.compile(r"\x1b\[((\d+;)*\d+)m")

    def insert_ansi(self, txt, index="insert"):
        first_line, first_char = map(
            int, str(self.textbox.index(index)).split("."))
        if index == "end":
            first_line -= 1

        lines = txt.splitlines()
        if not lines:
            return
        # insert text without ansi codes
        self.textbox.insert(index, self.ansi_regexp.sub('', txt))
        # find all ansi codes in txt and apply corresponding tags
        opened_tags = {}  # we need to keep track of the opened tags to be able to do
        # self.textbox.tag_add(tag, start, end) when we reach a "closing" ansi code

        def apply_formatting(code, code_index):
            if code == 0:  # reset all by closing all opened tag
                for tag, start in opened_tags.items():
                    self.textbox.tag_add(tag, start, code_index)
                opened_tags.clear()
            elif code in self.ansi_font_format:  # open font formatting tag
                tag = self.ansi_font_format[code]
                opened_tags[tag] = code_index
            elif code in self.ansi_font_reset:   # close font formatting tag
                tag = self.ansi_font_reset[code]
                if tag in opened_tags:
                    self.textbox.tag_add(tag, opened_tags[tag], code_index)
                    del opened_tags[tag]
            # open foreground color tag (and close previously opened one if any)
            elif code in self.ansi_color_fg:
                for tag in list(opened_tags):
                    if tag.startswith('foreground'):
                        self.textbox.tag_add(tag, opened_tags[tag], code_index)
                        del opened_tags[tag]
                opened_tags[self.ansi_color_fg[code]] = code_index
            # open background color tag (and close previously opened one if any)
            elif code in self.ansi_color_bg:
                for tag in list(opened_tags):
                    if tag.startswith('background'):
                        self.textbox.tag_add(tag, opened_tags[tag], code_index)
                        del opened_tags[tag]
                opened_tags[self.ansi_color_bg[code]] = code_index

        def find_ansi(line_txt, line_nb, char_offset):
            # difference between the character position in the original line and in the text widget
            delta = -char_offset
            # (initial offset due to insertion position if first line + extra offset due to deletion of ansi codes)
            for match in self.ansi_regexp.finditer(line_txt):
                codes = [int(c) for c in match.groups()[0].split(';')]
                start, end = match.span()
                for code in codes:
                    apply_formatting(code, "{}.{}".format(
                        line_nb, start - delta))
                delta += end - start  # take into account offset due to deletion of ansi code

        # first line, with initial offset due to insertion position
        find_ansi(lines[0], first_line, first_char)
        for line_nb, line in enumerate(lines[1:], first_line + 1):
            find_ansi(line, line_nb, 0)   # next lines, no offset
        # close still opened tag
        for tag, start in opened_tags.items():
            self.textbox.tag_add(tag, start, "end")

    def redirector(self, inputStr):
        self.textbox.configure(state='normal')
        self.textbox.mark_set("insert", "end")
        while '\b' in inputStr:
            bs_index = inputStr.index('\b')
            if bs_index == 0:
                # Delete the character before the current 'insert' position in the Text widget
                self.textbox.delete("insert-2c")
                inputStr = inputStr[1:]
            else:
                # Remove the character before the backspace and the backspace itself
                inputStr = inputStr[:bs_index-1] + inputStr[bs_index+1:]
        self.insert_ansi(inputStr, tk.INSERT)
        # self.textbox.insert(tk.INSERT, inputStr)
        self.textbox.configure(state='disabled')
        self.textbox.see(tk.END)  # Auto-scroll to the end
        self.old_stdout(inputStr)

    def on_resize(self, event):
        width = self.textbox.winfo_width()
        font_width = self.textbox.tk.call(
            "font", "measure", self.textbox["font"], "-displayof", self.textbox, "0")
        sys.stdout.write = self.redirector

    def check_queue(self):
        while not self.queue.empty():
            try:
                msg = self.queue.get_nowait()
                self.redirector(msg)
            except queue.Empty:
                break
        self.after(100, self.check_queue)

    def __del__(self):
        sys.stdout.write = self.old_stdout
