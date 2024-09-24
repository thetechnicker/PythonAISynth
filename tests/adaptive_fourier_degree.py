from datetime import datetime
import glob
import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# Constants
SAMPLE_RATE = 44100
INITIAL_FOURIER_DEGREE = 250
MIN_FOURIER_DEGREE = 1
MAX_FOURIER_DEGREE = 500
BATCH_SIZE = 100
NUM_EPOCHS = 10000
PATIENCE = 75
MIN_DIFF = 0.001
LEARNING_RATE = 0.01
EARLY_STOP_LOSS_THRESHOLD = 0.05
FUNCTION_NAME = 'sin'
EPOCHS_BEFORE_ADJUSTMENT = 100
TRAIN_SAMPLES = 1000

# autopep8: off
from context import scr
from scr import predefined_functions
# autopep8: on


class FourierLayer(nn.Module):
    def __init__(self, fourier_degree):
        super(FourierLayer, self).__init__()
        self.fourier_degree = fourier_degree
        self.tensor = torch.arange(1, fourier_degree + 1)

    def forward(self, x):
        self.tensor = self.tensor.to(x.device)
        y = x * self.tensor
        z = torch.cat((torch.sin(y), torch.cos(y)), dim=1)
        return z


class CustomLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super(CustomLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = nn.Parameter(torch.randn(out_features))

    def forward(self, x):
        return x @ self.weight.t() + self.bias

    def expand(self, new_in_features):
        new_weight = nn.Parameter(torch.randn(
            self.out_features, new_in_features))
        with torch.no_grad():
            new_weight[:, :self.in_features] = self.weight
        self.weight = new_weight
        self.in_features = new_in_features

    def reduce(self, new_in_features):
        new_weight = nn.Parameter(torch.randn(
            self.out_features, new_in_features))
        with torch.no_grad():
            new_weight[:] = self.weight[:, :new_in_features]
        self.weight = new_weight
        self.in_features = new_in_features


class SimpleModel(nn.Module):
    def __init__(self, fourier_degree):
        super(SimpleModel, self).__init__()
        self.fourier_degree = fourier_degree
        self.fourier_layer = FourierLayer(fourier_degree)
        self.linear_layer = CustomLinear(fourier_degree * 2, 1)

    def forward(self, x):
        fourier_features = self.fourier_layer(x)
        return self.linear_layer(fourier_features)

    def update_fourier_degree(self, new_degree):
        if self.fourier_degree < new_degree:
            self.fourier_degree = new_degree
            self.fourier_layer = FourierLayer(new_degree)
            self.linear_layer.expand(new_degree * 2)
        else:
            self.fourier_degree = new_degree
            self.fourier_layer = FourierLayer(new_degree)
            self.linear_layer.reduce(new_degree * 2)


def train_model(model, optimizer, criterion, train_loader, val_loader, num_epochs, min_degree, max_degree, patience=PATIENCE):
    lowest_loss = torch.inf
    lowest_loss_degree = model.fourier_degree
    epoch = 0
    epochs_without_improvement = 0

    while epoch < num_epochs:
        model.train()
        train_loss = 0.0
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)

        # Validation step
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()

        val_loss /= len(val_loader)

        print(f"Epoch {(epoch + 1):10}/{num_epochs:10}, Train Loss: {train_loss:10.5f}, Val Loss: {val_loss:10.5f}, Lowest Loss: {lowest_loss:10.5f} ({lowest_loss_degree:3}), Fourier Degree: {model.fourier_degree:10}")

        # Dynamic adjustment of fourier_degree
        if train_loss < lowest_loss:
            lowest_loss = train_loss
            lowest_loss_degree = model.fourier_degree
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        # Adjust Fourier degree based on loss
        if (epoch + 1) % EPOCHS_BEFORE_ADJUSTMENT == 0:
            if abs(lowest_loss - train_loss) < MIN_DIFF:
                if model.fourier_degree < max_degree:
                    model.update_fourier_degree(model.fourier_degree + 1)
                    optimizer = optim.Adam(
                        model.parameters(), lr=LEARNING_RATE)
                    print(
                        f"Increasing Fourier degree to {model.fourier_degree}")
            elif train_loss > lowest_loss + EARLY_STOP_LOSS_THRESHOLD:
                if model.fourier_degree > min_degree:
                    model.update_fourier_degree(model.fourier_degree - 1)
                    optimizer = optim.Adam(
                        model.parameters(), lr=LEARNING_RATE)
                    print(
                        f"Decreasing Fourier degree to {model.fourier_degree}")

        # Check for early stopping condition
        if epochs_without_improvement >= patience or val_loss < 0.0002:
            print(f"Early stopping triggered after {epoch + 1} epochs.")
            break

        epoch += 1


class DualOutput:
    def __init__(self, filename):
        self.file = open(filename, 'a')

    def write(self, message):
        self.file.write(message)
        sys.__stdout__.write(message)

    def flush(self):
        self.file.flush()

    def close(self):
        self.file.close()


def filter_and_keep_newest_files(folder_path, identifier_str, num_files_to_keep=10):
    # Create the search pattern
    search_pattern = os.path.join(folder_path, f"*{identifier_str}.txt")

    # Get a list of files matching the pattern
    matching_files = glob.glob(search_pattern)

    # Sort the files by modification time (newest first)
    matching_files.sort(key=os.path.getmtime, reverse=True)

    # Keep only the newest files
    files_to_keep = matching_files[:num_files_to_keep]

    # Determine which files to delete
    files_to_delete = matching_files[num_files_to_keep:]

    # Print the files to keep and delete (for verification)
    print("Keeping the following files:")
    for file in files_to_keep:
        print(file)

    print("\nDeleting the following files:")
    for file in files_to_delete:
        print(file)
        os.remove(file)  # Uncomment this line to actually delete the files


if __name__ == "__main__":
    filter_and_keep_newest_files("tests/tmp/", "_log_12345")

    timestamp = datetime.now()
    dual_output = DualOutput(
        f'tests/tmp/{timestamp.strftime("%Y-%m-%d-%H:%M")}_log_12345.txt')
    sys.stdout = dual_output

    # Create a simple dataset
    inputs = torch.linspace(-torch.pi, torch.pi, TRAIN_SAMPLES).unsqueeze(1)
    targets = torch.tensor(
        [predefined_functions.predefined_functions_dict[FUNCTION_NAME](x) for x in inputs], dtype=torch.float32).unsqueeze(1)

    dataset = torch.utils.data.TensorDataset(inputs, targets)
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE)
    val_loader = torch.utils.data.DataLoader(
        dataset, batch_size=BATCH_SIZE)

    model = SimpleModel(INITIAL_FOURIER_DEGREE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()

    # Train the model with dynamic degree adjustment
    train_model(model, optimizer, criterion, train_loader,
                val_loader, num_epochs=NUM_EPOCHS, min_degree=MIN_FOURIER_DEGREE, max_degree=MAX_FOURIER_DEGREE)

    t2 = 2 * np.pi * np.linspace(0, 1, SAMPLE_RATE) - np.pi
    t2_tensor = torch.tensor(t2, dtype=torch.float32).unsqueeze(1)
    with torch.no_grad():
        plt.plot(inputs.cpu().numpy(), targets.cpu().numpy(), label="stupid")
        plt.plot(t2, model(t2_tensor).cpu().numpy(), label=f"model")
        # plt.plot(
        #     t2, [predefined_functions.predefined_functions_dict[FUNCTION_NAME](x) for x in (t2+np.pi)], label="actual")

    plt.xlabel('x (radians)')
    plt.ylabel('sin(x)')
    plt.title('Sine Wave')
    plt.grid(True)
    plt.legend()
    plt.show()

    sys.stdout.close()
    sys.stdout = sys.__stdout__
