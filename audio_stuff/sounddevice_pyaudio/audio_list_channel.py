import pyaudio
p = pyaudio.PyAudio()


for i in range(p.get_device_count()):
    info = p.get_device_info_by_index(i)
    print("Device id ", i, " - ", info["name"].ljust(35),
          " - ", "Max input channel:", info["maxInputChannels"],
          " - ", "Max output channel:", info["maxOutputChannels"])

