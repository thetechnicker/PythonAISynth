import mido
import time

# Open the Microsoft GS Wavetable Synth for output
midiout = mido.open_output('Microsoft GS Wavetable Synth 0')

# Create a 'note_on' message
on_msg = mido.Message('note_on', note=60, velocity=127)

# Send the 'note_on' message
midiout.send(on_msg)

# Wait for 5 seconds
time.sleep(5)

# Create a 'note_off' message
off_msg = mido.Message('note_off', note=60)

# Send the 'note_off' message
midiout.send(off_msg)
