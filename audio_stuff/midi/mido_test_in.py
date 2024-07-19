import mido


def listen_midi():
    # Open a virtual port for input
    midiin = mido.open_input('TEST', virtual=True)

    while True:
        # Wait for a message and print it
        msg = midiin.receive()
        print(msg)


if __name__ == "__main__":
    listen_midi()
