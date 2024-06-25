import atexit
import multiprocessing
from multiprocessing.pool import IMapIterator, MapResult
import os
import signal
import sys
import threading
import time
import tkinter as tk
from tkinter import filedialog, ttk
import numpy as np
import pretty_midi
import sounddevice as sd
from scipy.io.wavfile import write

from scr import utils
from scr.fourier_neural_network import FourierNN
from multiprocessing import Manager, Pool, Process


def musik_from_file(fourier_nn: FourierNN):
    midi_file = filedialog.askopenfilename()
    if not midi_file:
        return
    print(midi_file)
    midi_data = pretty_midi.PrettyMIDI(midi_file)
    # print(dir(midi_data))
    notes_list = []
    # Iterate over all instruments and notes in the MIDI file
    if fourier_nn:
        for instrument in midi_data.instruments:
            # print(dir(instrument))
            for note_a, note_b in utils.note_iterator(instrument.notes):
                # node_a
                print(note_a)
                duration = note_a.end - note_a.start
                synthesized_note = fourier_nn.synthesize_2(
                    midi_note=note_a.pitch-12, duration=duration, sample_rate=44100)
                print(synthesized_note.shape)
                notes_list.append(synthesized_note)

                if note_b:
                    # pause
                    print("Pause")
                    duration = abs(note_b.start - note_a.end)
                    synthesized_note=np.zeros((int(duration*44100), 1))
                    print(synthesized_note.shape)
                    notes_list.append(synthesized_note)

                # # node_b
                # print(note_b)
                # duration = note_b.end - note_b.start
                # synthesized_note = fourier_nn.synthesize_2(
                #     midi_note=note_b.pitch, duration=duration, sample_rate=44100)
                # print(synthesized_note.shape)
                # notes_list.append(synthesized_note)
            break
    output = np.concatenate(notes_list)
    sd.play(output, 44100)



try:
    import mido
    mido.open_input('TEST', virtual=True).close()
    proc=None

    def wrapper(func, param):
        print(os.getcwd())
        sys.stdout = open(f'tmp/process_{os.getpid()}_output.txt', 'w')
        print("test")
        return func(*param)


    def midi_proc(fourier_nn:FourierNN):
        blink=True
        notes:dict=None
        note_list:list=[]
        finished=0
        with Pool(os.cpu_count()) as pool:
            results = pool.imap_unordered(fourier_nn.synthesize_3, range(128))

            while True:
                try:
                    result = results.next(1)
                    note_list.append(result)
                    finished+=1
                except multiprocessing.TimeoutError:
                    pass
                except StopIteration:
                    break
                if blink:
                    print(f"{finished}/128 finished. pennding".ljust(100, " "), end="\r")
                    blink=False
                else:
                    print(f"{finished}/128 finished.".ljust(100, " "), end="\r")
                    blink=True
                   
            notes=dict(note_list)
        print("Ready")

        with mido.open_input('TEST', virtual=True) as midiin:
            # y=np.zeros((44100, 1,))
            while True:
                for msg in midiin.iter_pending():
                    print(msg)
                    if msg.type == 'note_off':
                        sd.stop()
                    elif msg.type == 'note_on':
                        sd.play(notes[msg.note], 44100, blocking=False)

    def midi_to_musik_live(root, fourier_nn:FourierNN):
        global proc
        if proc:
            utils.DIE(proc)
            proc=None
                
        proc=Process(target=midi_proc, args=(fourier_nn,))
        proc.start()
        atexit.register(utils.DIE, proc)

except:
    pass