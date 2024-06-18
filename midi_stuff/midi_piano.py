import tkinter as tk
import mido

# Create a new Tkinter window
window = tk.Tk()

# Create a variable to store the selected MIDI port
selected_port = tk.StringVar(window)
selected_port.set(mido.get_output_names()[0])  # default value

# Create a dropdown menu to select the MIDI port
port_menu = tk.OptionMenu(window, selected_port, *mido.get_output_names())
port_menu.pack()

# Create a variable to store the selected MIDI channel
selected_channel = tk.StringVar(window)
selected_channel.set('1')  # default value

# Create a dropdown menu to select the MIDI channel
channel_menu = tk.OptionMenu(window, selected_channel, *[str(i) for i in range(1, 17)])
channel_menu.pack()

# Function to send MIDI messages
def send_midi_note(note, velocity=64, type='note_on'):
    port = mido.open_output(selected_port.get())
    channel = int(selected_channel.get()) - 1
    msg = mido.Message(type, note=note, velocity=velocity, channel=channel)
    port.send(msg)

# Create piano keys
for i in range(21, 109):  # MIDI notes 21 (A0) to 108 (C8)
    button = tk.Button(window, text=str(i))
    button.pack(side=tk.LEFT)
    button.bind('<ButtonPress-1>', lambda event, i=i: send_midi_note(i, type='note_on'))
    button.bind('<ButtonRelease-1>', lambda event, i=i: send_midi_note(i, type='note_off'))

# Run the Tkinter event loop
window.mainloop()
