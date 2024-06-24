import atexit
import os
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


# Define your long running tasks here
def long_running_task(args, list):
    func, params = args
    result = func(*params)  # Execute the function with the given parameters
    list.append(result)  # Append the result to the list when the task is done

def update_progressbar(parent, list, progressbar, total_tasks):
    while True:
        progressbar['value'] = len(list)  # Update the progress bar
        if progressbar['value'] < total_tasks:
            parent.after(100, update_progressbar, parent, list, progressbar, total_tasks)
            

def execute_with_progressbar(parent, func, iterable):
    total_tasks = len(iterable)

    # Create a new top-level window
    progress_window = tk.Toplevel(parent)
    progress_window.title("Progress")

    progressbar = ttk.Progressbar(progress_window, length=300, mode='determinate', maximum=total_tasks)
    progressbar.pack()

    with Manager() as manager:
        list = manager.list()  # Shared list

        # Start the long running tasks
        with Pool(processes=os.cpu_count()) as pool:
            pool.starmap_async(long_running_task, [((func, params), list) for params in iterable])

        # Start a loop that updates the progress bar
        progress_window.after(100, update_progressbar, progress_window, list, progressbar, total_tasks)

    # Wait for all tasks to complete
    while len(list) < total_tasks:
        time.sleep(0.1)

    # At this point, all tasks have completed and their results are in the list
    results = list[:]

    progress_window.quit()

    return results

try:
    import mido
    mido.open_input('TEST', virtual=True).close()
    proc=None


    def midi_proc(notes):
        print("Ready")
        with mido.open_input('TEST', virtual=True) as midiin:
            # y=np.zeros((44100, 1,))
            while True:
                for msg in midiin.iter_pending():
                    print(msg)
                    if msg.type == 'note_off':
                        sd.stop()
    
                    sd.play(notes[msg.note], 44100, blocking=False)
    
    # def synth(note, fourier_nn:FourierNN):
    #     duration=1
    #     sample_rate=44100
    #     freq = utils.midi_to_freq(note)
    #     t = np.linspace(0, duration, int(sample_rate * duration), False)
    #     t_scaled = t * 2 * np.pi / (1/freq)
    #     output = fourier_nn.predict(t_scaled)
    #     output = output / np.max(np.abs(output))  # Normalize
    #     return (note, output.astype(np.int16))

    def midi_to_musik_live(root, fourier_nn:FourierNN):
        global proc
        if proc:
            utils.DIE(proc)
            proc=None
            
        # notes = execute_with_progressbar(root, fourier_nn.synthesize_3, range(128))
        with Pool(4) as pool:
            notes=pool.map(fourier_nn.synthesize_3, range(128))

        proc=Process(target=midi_proc, args=[dict(notes)])
        proc.start()
        atexit.register(utils.DIE, proc)
except:
    pass