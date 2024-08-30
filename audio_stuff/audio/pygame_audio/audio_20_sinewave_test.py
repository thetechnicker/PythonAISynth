import random
import pygame
import numpy as np

# Initialize Pygame
pygame.init()

# Set up the mixer with a frequency of 44100 Hz, 16-bit depth, 2 channels, and a buffer of 1024 samples
pygame.mixer.init(frequency=44100, size=-16, channels=1, buffer=1024)

# Set the number of channels to 20
pygame.mixer.set_num_channels(20)

# Function to generate a sine wave


def generate_sine_wave(frequency, duration, sample_rate=44100, amplitude=4096):
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    sine_wave = amplitude * np.sin(frequency * t * 2 * np.pi)
    return sine_wave.astype(np.int16)


# Generate and play 20 sine waves
for i in range(20):
    frequency = 100+100 * (i + 1)  # Frequency in Hz
    duration = random.randint(100, 500)/100  # Duration in seconds

    # Generate the sine wave
    sine_wave = generate_sine_wave(frequency, duration)

    # Convert to stereo (2 channels)
    stereo_sine_wave = np.repeat(sine_wave.reshape(-1, 1), 2, axis=1)

    # Create a Pygame Sound object
    sound = pygame.sndarray.make_sound(stereo_sine_wave)

    # Play the sound on a specific channel
    channel = pygame.mixer.Channel(i)
    channel.play(sound)

channel_running: list[bool] = [True]*20
try:
    while any(channel_running):
        for i in range(20):
            if not pygame.mixer.Channel(i).get_busy():
                channel_running[i] = False
        print(f"{sum(channel_running)}/20 still running")
except KeyboardInterrupt:
    pass
