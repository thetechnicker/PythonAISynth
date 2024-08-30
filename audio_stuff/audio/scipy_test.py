import tkinter as tk
import numpy as np
from scipy.signal import lfilter, butter
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
# import sounddevice as sd


class EnvelopeGraph(tk.Frame):
    def __init__(self, master=None, **kwargs):
        super().__init__(master, **kwargs)
        self.envelope = None
        self.previous_envelope = None
        self.grid(sticky='NSEW')
        self.rowconfigure(1, weight=1)
        for i in range(5):
            self.columnconfigure(i, weight=1)

        # Create a new figure and a subplot
        self.fig = Figure(figsize=(5, 5), dpi=100)
        self.ax = self.fig.add_subplot(1, 1, 1)

        # Create a canvas and add it to the window
        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        self.canvas.get_tk_widget().grid(row=0, column=0, columnspan=5, sticky='NSEW')

        # Create Scale widgets for the ADSR parameters
        self.attack_scale = self.create_scale('Attack')
        self.decay_scale = self.create_scale('Decay')
        self.sustain_scale = self.create_scale('Sustain')
        self.sustain_scale_2 = self.create_scale('Sustain length')
        self.release_scale = self.create_scale('Release')

        # Set the initial values of the Scale widgets
        self.attack_scale.set(0.1)
        self.decay_scale.set(0.1)
        self.sustain_scale.set(0.5)
        self.release_scale.set(0.2)
        self.sustain_scale_2.set(0.6)

        # Grid the Scale widgets
        self.attack_scale.grid(row=1, column=0, sticky='NSEW')
        self.decay_scale.grid(row=1, column=1, sticky='NSEW')
        self.sustain_scale.grid(row=1, column=2, sticky='NSEW')
        self.sustain_scale_2.grid(row=1, column=3, sticky='NSEW')
        self.release_scale.grid(row=1, column=4, sticky='NSEW')

    def create_scale(self, label):
        scale = tk.Scale(self, from_=0, to=1, resolution=0.01,
                         orient='horizontal', label=label, command=self.update_graph)
        return scale

    def update_graph(self, *args):
        # Get the ADSR parameters from the Scale widgets
        attack_time = self.attack_scale.get()
        decay_time = self.decay_scale.get()
        release_time = self.release_scale.get()
        sustain_time = self.sustain_scale_2.get()
        total_time = attack_time+decay_time+release_time+sustain_time
        total_time = total_time if total_time > 0 else 1
        attack_time = attack_time/total_time
        decay_time = decay_time/total_time
        release_time = release_time/total_time
        sustain_time = sustain_time/total_time
        sustain_level = self.sustain_scale.get()

        # Generate the sound sample and envelope
        fs = 44100
        f = 440
        t = np.arange(fs) / fs
        self.sample = np.sin(2 * np.pi * f * t)

        attack_samples = int(attack_time * fs)
        decay_samples = int(decay_time * fs)
        release_samples = int(release_time * fs)
        sustain_samples = int(sustain_time * fs)
        fix = attack_samples+decay_samples+release_samples+sustain_samples
        sustain_samples += fs-fix
        try:
            self.envelope = np.concatenate([
                np.linspace(0, 1, attack_samples),  # Attack
                np.linspace(1, sustain_level, decay_samples),  # Decay
                np.full(sustain_samples, sustain_level),  # Sustain
                np.linspace(sustain_level, 0, release_samples)  # Release
            ])

            # Clear the subplot
            self.ax.clear()

            # Plot the envelope
            self.ax.plot(t, self.envelope)
            self.ax.set_title('Envelope')
            self.ax.set_xlabel('Time (s)')
            self.ax.set_ylabel('Amplitude')

            # Redraw the canvas
            self.canvas.draw()

        except:
            pass


# Create a new Tkinter window
window = tk.Tk()
window.rowconfigure(1, weight=1)
window.columnconfigure(0, weight=1)

# Add the EnvelopeGraph frame to the window
graph = EnvelopeGraph(window)
graph.grid(sticky='NSEW')

# Run the Tkinter event loop
window.mainloop()
