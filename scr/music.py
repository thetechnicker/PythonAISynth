import atexit
import multiprocessing
from multiprocessing.pool import IMapIterator, MapResult
from multiprocessing import Manager, Pool, Process
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

    def init_worker():
        # Ignore the SIGINT signal in worker processes, to handle it in the parent
        signal.signal(signal.SIGINT, signal.SIG_IGN)

    def midi_proc(fourier_nn:FourierNN):
        blink=True
        notes:dict=None
        note_list:list=[]
        finished=0
        original_sigint_handler = signal.signal(signal.SIGINT, signal.SIG_IGN)
        try:
            with multiprocessing.Pool(initializer=init_worker, processes=os.cpu_count()) as pool:
                signal.signal(signal.SIGINT, original_sigint_handler)
                note_list = pool.map(fourier_nn.synthesize_3, range(21,128))
        except KeyboardInterrupt:
            print("Caught KeyboardInterrupt, terminating workers")
            pool.terminate()
            pool.join()
            return
        
        notes=dict(note_list)
        # with Pool(processes=os.cpu_count()) as pool:
        #     note_list = pool.map(fourier_nn.synthesize_3, range(21,128))
        #     # results = pool.imap_unordered(fourier_nn.synthesize_3, range(128))

        #     # while True:
        #     #     try:
        #     #         result = results.next(1)
        #     #         note_list.append(result)
        #     #         finished+=1
        #     #     except multiprocessing.TimeoutError:
        #     #         pass
        #     #     except StopIteration:
        #     #         break
        #     #     if blink:
        #     #         print(f"{finished}/128 finished. pennding".ljust(100, " "))
        #     #         blink=False
        #     #     else:
        #     #         print(f"{finished}/128 finished.".ljust(100, " "))
        #     #         blink=True
        #     #     print("#################################", len(note_list))

        #     notes=dict(note_list)
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
                        active_notes = {}

    active_notes = {}

    def midi_proc_2(fourier_nn:FourierNN):
        with mido.open_input('TEST', virtual=True) as midiin:
            print("Ready")

            fs = 44100  # Sample rate
            f1 = 440  # Frequency of the "duuu" sound (in Hz)
            f2 = 880  # Frequency of the "dib" sound (in Hz)
            t1 = 0.8  # Duration of the "duuu" sound (in seconds)
            t2 = 0.2  # Duration of the "dib" sound (in seconds)
            t3 = 0.1

            # Generate the "duuu" sound
            t = np.arange(int(t1 * fs)) / fs
            sound1 = 0.5 * np.sin(2 * np.pi * f1 * t)

            # Generate the "dib" sound
            t = np.arange(int(t2 * fs)) / fs
            sound2 = 0.5 * np.sin(2 * np.pi * f2 * t)

            # Concatenate the two sounds
            sounds = np.concatenate([sound1, sound2])
            sd.play(sounds, fs, blocking=True)
            sounds=np.zeros(shape=(fs*t3, 1))
            while True:
                i=0
                for msg in midiin.iter_pending():
                    print("new msg", msg, i)
                    if msg.type == 'note_off':
                        # When a note is released, stop playing it
                        if msg.note in active_notes:
                            del active_notes[msg.note]
                            sd.stop()
                    elif msg.type == 'note_on':
                        # When a note is pressed, start playing it
                        active_notes[msg.note] = fourier_nn.synthesize_2(msg.note, t3)
                    
                    for note, sound in list(active_notes.items()):
                        sounds += sound

                    sounds = (sounds * 32767 / np.max(np.abs(sounds))) / 2

                    sd.play(sounds, fs, blocking=False)
                    
                    i+=1

                # Sleep for a while to reduce CPU usage
                time.sleep(0.01)
        


    def midi_to_musik_live(root, fourier_nn:FourierNN):
        global proc
        if proc:
            utils.DIE(proc)
            proc=None
        fourier_nn.save_tmp_model()
        proc=Process(target=midi_proc_2, args=(fourier_nn,))
        proc.start()
        atexit.register(utils.DIE, proc)

except:
    pass
