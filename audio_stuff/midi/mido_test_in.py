import mido


def listen_midi():
    # Open a virtual port for input
    default_name = mido.get_input_names()[0]
    with mido.open_input(default_name) as midi_in:  # 'Digital Piano 2')
        i = 0
        while True:
            for msg in midi_in.iter_pending():
                print()
            # print(i)
            # i += 1


if __name__ == "__main__":
    listen_midi()
