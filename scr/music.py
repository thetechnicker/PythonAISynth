import atexit
import multiprocessing
from multiprocessing import Process
import os
import signal
import sys
from tkinter import filedialog
import numpy as np
import pretty_midi
import sounddevice as sd

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



if ((not os.getenv('HAS_RUN_INIT')) or os.getenv('play')=='true'):
    try:
        import pygame
        import mido
        port_name='AI_SYNTH'
        virtual=True
        if not os.getenv('HAS_RUN_INIT'):
            try:
                mido.open_input(port_name, virtual=virtual).close()
            except NotImplementedError:
                print("test loopBe midi")
                port_name="LoopBe Internal MIDI 0"
                virtual=False
                mido.open_input(port_name, virtual=virtual).close()
                print("loopbe works")
        proc=None

        def wrapper(func, param):
            print(os.getcwd())
            sys.stdout = open(f'tmp/process_{os.getpid()}_output.txt', 'w')
            print("test")
            return func(*param)

        def init_worker():
            # Ignore the SIGINT signal in worker processes, to handle it in the parent
            signal.signal(signal.SIGINT, signal.SIG_IGN)

        
        def midi_proc(fourier_nn:FourierNN, virtual:bool):
            pygame.init()
            pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=1024)
            os.environ['play']='true'
            notes:dict={}
            note_list:list=[]
            original_sigint_handler = signal.signal(signal.SIGINT, signal.SIG_IGN)
            try:
                with multiprocessing.Pool(initializer=init_worker, processes=os.cpu_count()) as pool:
                    signal.signal(signal.SIGINT, handler=original_sigint_handler)
                    note_list = pool.map(fourier_nn.synthesize_3, range(128))
            except KeyboardInterrupt:
                print("Caught KeyboardInterrupt, terminating workers")
                pool.terminate()
                pool.join()
                return
            # note_list=[create_pygame_sound(fourier_nn, i)  for i in range(128)]
            for note, sound in note_list:
                stereo = np.repeat(sound.reshape(-1, 1), 2, axis=1)
                stereo_sound = pygame.sndarray.make_sound(stereo)
                notes[note]=stereo_sound

            print(notes.keys())

            # Set the number of channels to 20
            pygame.mixer.set_num_channels(20)



            fs = 44100  # Sample rate
            f1 = 440  # Frequency of the "duuu" sound (in Hz)
            f2 = 880  # Frequency of the "dib" sound (in Hz)
            t1 = 0.8  # Duration of the "duuu" sound (in seconds)
            t2 = 0.2  # Duration of the "dib" sound (in seconds)

            # Generate the "duuu" sound
            t = np.arange(int(t1 * fs)) / fs
            sound1 = 0.5 * np.sin(2 * np.pi * f1 * t)

            # Generate the "dib" sound
            t = np.arange(int(t2 * fs)) / fs
            sound2 = 0.5 * np.sin(2 * np.pi * f2 * t)

            # Concatenate the two sounds
            audio = np.concatenate([sound1, sound2])
            output = np.array(audio * 32767 / np.max(np.abs(audio)) / 2).astype(np.int16)
            stereo_sine_wave = np.repeat(output.reshape(-1, 1), 2, axis=1)
            sound = pygame.sndarray.make_sound(stereo_sine_wave)

            # Play the sound on a specific channel
            running_channels:dict[str, tuple[int, pygame.mixer.Channel]] = {}
            free_channel_ids=list(range(20))
            channel=pygame.mixer.Channel(0)
            channel.play(sound)
            while(channel.get_busy()):
                pass
            print("Ready")
            with mido.open_input(port_name, virtual=virtual) as midiin:
                while True:
                    for msg in midiin.iter_pending():
                        print(msg)
                        if msg.type == 'note_off':
                            free_channel_ids.append(running_channels[notes[msg.note]][0])
                            running_channels[notes[msg.note]][1].stop()
                        elif msg.type == 'note_on':
                            id=free_channel_ids.pop()
                            channel=pygame.mixer.Channel()
                            channel.play(notes[msg.note])
                            running_channels[notes[msg.note]]=(id, channel,)


        def midi_to_musik_live(root, fourier_nn:FourierNN):
            os.environ['play']='true'
            global proc
            if proc:
                utils.DIE(proc)
                proc=None
            fourier_nn.save_tmp_model()
            proc=Process(target=midi_proc, args=(fourier_nn,virtual,))
            proc.start()
            atexit.register(utils.DIE, proc)
    
        if not os.getenv('HAS_RUN_INIT'):
            print("live music possible")
    except Exception as e:
        print("live music NOT possible", e, sep="\n")
