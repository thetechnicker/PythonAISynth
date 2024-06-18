import time
import tkinter as tk
import mido
import locale

octaves_to_display = 2
# Create a new Tkinter window
window = tk.Tk()

# Create a variable to store the selected MIDI port
selected_port = tk.StringVar(window)
selected_port.set(mido.get_output_names()[0])  # default value
# print(mido.get_output_names()[0])#Microsoft GS Wavetable Synth 0
port = mido.open_output(selected_port.get())


def changePort(event):
    global port
    port.close()
    port = mido.open_output(selected_port.get())


# Create a dropdown menu to select the MIDI port
port_menu = tk.OptionMenu(window, selected_port, *
                          mido.get_output_names(), command=changePort)
port_menu.pack()

# Create a variable to store the selected MIDI channel
selected_channel = tk.StringVar(window)
selected_channel.set('1')  # default value

# Create a dropdown menu to select the MIDI channel
channel_menu = tk.OptionMenu(
    window, selected_channel, *[str(i) for i in range(1, 17)])
channel_menu.pack()

# Create a variable to store the selected octave
selected_octave = tk.IntVar(window)
selected_octave.set(4)  # default value

# Create a slider to select the octave
octave_slider = tk.Scale(window, from_=0, to=8, orient='horizontal',
                         variable=selected_octave, label='Octave')
octave_slider.pack()

# Function to send MIDI messages

note_states = {i: False for i in range(octaves_to_display*12)}


def send_midi_note(note, velocity=127, type='note_on'):
    global note_states

    if type == 'note_on':
        # If the note is already on, don't send another note_on message
        if note_states[note]:
            return
        else:
            note_states[note] = True
    elif type == 'note_off':
        note_states[note] = False
        # time.sleep(0.3)

    channel = int(selected_channel.get()) - 1
    msg = mido.Message(type, note=note + selected_octave.get()
                       * 12, velocity=velocity, channel=channel)
    port.send(msg)


# Create piano keys
keys_frame = tk.Frame(window)
keys_frame.pack()

white_keys = [0, 2, 4, 5, 7, 9, 11]  # , 12]
black_keys = [1, 3, 6, 8, 10]
# Define your keys for white and black buttons
white_keys_bindings = ['a', 's', 'd', 'f', 'g', 'h', 'j']
black_keys_bindings = ['w', 'e', 'r', 't', 'y']

for i in range(7):
    button = tk.Button(keys_frame, text=' ', bg='white', width=2, height=6)
    button.grid(row=0, column=i*2)
    button.bind('<ButtonPress-1>', lambda event,
                i=i: send_midi_note(white_keys[i], type='note_on'))
    button.bind('<ButtonRelease-1>', lambda event,
                i=i: send_midi_note(white_keys[i], type='note_off'))
    # Bind the button to a key press event
    window.bind('<KeyPress-%s>' % white_keys_bindings[i], lambda event,
                i=i: send_midi_note(white_keys[i], type='note_on'))
    window.bind('<KeyRelease-%s>' % white_keys_bindings[i], lambda event,
                i=i: send_midi_note(white_keys[i], type='note_off'))

for i in range(5):
    button = tk.Button(keys_frame, text=' ', bg='black', width=1, height=4)
    button.grid(row=0, column=black_keys[i]*2-1, sticky='n')
    button.bind('<ButtonPress-1>', lambda event,
                i=i: send_midi_note(black_keys[i], type='note_on'))
    button.bind('<ButtonRelease-1>', lambda event,
                i=i: send_midi_note(black_keys[i], type='note_off'))
    # Bind the button to a key press event
    window.bind('<KeyPress-%s>' % black_keys_bindings[i], lambda event,
                i=i: send_midi_note(black_keys[i], type='note_on'))
    window.bind('<KeyRelease-%s>' % black_keys_bindings[i], lambda event,
                i=i: send_midi_note(black_keys[i], type='note_off'))


# Run the Tkinter event loop
window.mainloop()
