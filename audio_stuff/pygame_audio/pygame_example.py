import pygame

# Initialize Pygame
pygame.init()

# Initialize the mixer
pygame.mixer.init()

# Load a sound file
sound = pygame.mixer.Sound("/path/to/file")

# Play the sound
sound.play()

# Keep the program running long enough to hear the sound

pygame.time.wait(int(sound.get_length() * 1000))