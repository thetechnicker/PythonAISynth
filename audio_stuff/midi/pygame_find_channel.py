import pygame.midi

pygame.midi.init()


for i in range(pygame.midi.get_count()):
    info = pygame.midi.get_device_info(i)
    print(
        f"Device ID: {i}, Name: {info[1].decode()}, Input: {info[2]}, Output: {info[3]}")


desired_name = "Your MIDI Device Name"
device_id = None

for i in range(pygame.midi.get_count()):
    info = pygame.midi.get_device_info(i)
    print(info)
    if info[1].decode() == desired_name:
        device_id = i
        break

if device_id is not None:
    midi_output = pygame.midi.Output(device_id)
    print(f"Selected device ID: {device_id}")
else:
    print("Device not found")
