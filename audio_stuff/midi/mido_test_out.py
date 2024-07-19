import mido


def send_midi():
    # Open a virtual port for output
    midiout = mido.open_output('My virtual port', virtual=True)

    while True:
        # Ask the user for the note and velocity
        note = int(input("Enter the MIDI note number (0-127): "))
        velocity = int(input("Enter the velocity (0-127): "))

        # Create a MIDI message
        msg = mido.Message('note_on', note=note, velocity=velocity)

        # Send the message
        midiout.send(msg)


if __name__ == "__main__":
    send_midi()
