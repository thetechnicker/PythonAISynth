import mido
import time

# Define the MIDI output port
output = mido.open_output("LoopBe Internal MIDI 1")

# Define the notes for "Für Elise"
fur_elise = [
    ("E5", 0.5),
    ("D#5", 0.5),
    ("E5", 0.5),
    ("D#5", 0.5),
    ("E5", 0.5),
    ("B4", 0.5),
    ("D5", 0.5),
    ("C5", 0.5),
    ("A4", 1.5),
    ("C4", 0.5),
    ("E4", 0.5),
    ("A4", 0.5),
    ("B4", 1.5),
    ("E4", 0.5),
    ("G#4", 0.5),
    ("B4", 0.5),
    ("C5", 1.5),
    ("E4", 0.5),
    ("C4", 0.5),
    ("E4", 0.5),
    ("A4", 0.5),
    ("B4", 1.5),
    ("E4", 0.5),
    ("G#4", 0.5),
    ("B4", 0.5),
    ("C5", 1.5),
]

# Define a function to convert note names to MIDI note numbers


def note_to_midi(note):
    note_map = {
        "C": 0,
        "C#": 1,
        "D": 2,
        "D#": 3,
        "E": 4,
        "F": 5,
        "F#": 6,
        "G": 7,
        "G#": 8,
        "A": 9,
        "A#": 10,
        "B": 11,
    }
    octave = int(note[-1])
    key = note[:-1]
    return 12 * (octave + 1) + note_map[key]


try:
    # Stream the notes over MIDI in a loop
    last_note = None
    while True:
        for note, duration in fur_elise:
            midi_note = note_to_midi(note)
            last_note = midi_note
            output.send(mido.Message("note_on", note=midi_note, velocity=64))
            time.sleep(duration / 2)
            output.send(mido.Message("note_off", note=midi_note, velocity=64))
except:
    output.panic()
output.send(mido.Message("note_off", note=last_note, velocity=64))
# Close the MIDI output port (this line will never be reached due to the infinite loop)
# output.close()
