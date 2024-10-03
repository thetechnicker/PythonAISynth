import time
import mido
from mido import MidiFile, MidiTrack, Message


def play_channel(midi_file_path, channel_to_play):
    midi_file = MidiFile(midi_file_path)
    output = mido.open_output('LoopBe Internal MIDI 1')

    start_time = time.time()
    for track in midi_file.tracks:
        current_time = start_time
        for msg in track:
            if not msg.is_meta and not msg.type == 'sysex':
                if msg.channel == channel_to_play:
                    # Convert time from milliseconds to seconds
                    time.sleep(msg.time / 1000.0)
                    print(msg)
                    mido.Message(msg.type, note=msg.note,
                                 velocity=msg.velocity, channel=0)
                    output.send(msg)
                    current_time += msg.time / 1000.0


if __name__ == "__main__":
    midi_file_path = str(input("input filename: ")).strip()
    channel_to_play = int(input("Enter the channel number to play (0-15): "))
    play_channel(midi_file_path, channel_to_play)
