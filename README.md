# PythonAISynth

PythonAISynth is a software designed to create music using a neural network. The network uses a Fourier-Series based Regression model to approximate a curve on a Plot that a user can draw on. the model is then used to generate sound with higher sample rate than the drawing.

## Branches

- `main`: This branch contains the latest stable version.
- `dev`: This branch contains the current developments. It does not always run.

## Setup

To set up a Python environment for this project using Conda, run the following commands:

```bash
conda env create -f environment.yml -n ai_synth
conda activate ai_synth
```

## Usage

Once the project setup is complete, you can initiate it by executing the `main.py` file in the environment:

```bash
python -m pythonaisynth.main
```

![window](docs/img/main_window_v3.png)

Follow the steps below to generate sound:

1. Draw a curve on the graph.
2. Click on "Start Training". The neural network will start approximating this curve using a Fourier basis.
3. After the training is finished, click on "Play".
4. Now, send a MIDI signal to the program. It will generate sound based on the approximation of the curve.

Enjoy creating unique sounds with PythonAISynth! 🎵

## Contributing

If you have any suggestions or improvements for this project, please feel free to open an issue.

## License

This project is open source and licensed under the MIT License. Refer to the LICENSE file for more details.
