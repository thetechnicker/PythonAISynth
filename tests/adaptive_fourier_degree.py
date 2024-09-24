from datetime import datetime
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

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
        # print(x.shape, self.tensor.shape)
        y = x * self.tensor
        z = torch.cat((torch.sin(y), torch.cos(y)), dim=1)
        # print(z.shape)
        return z


class CustomLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super(CustomLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = nn.Parameter(torch.randn(out_features))

    def forward(self, x):
        # print(x.dtype, self.weight.t().dtype, self.bias.dtype)
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
            # Expand linear layer to match new Fourier features
            self.linear_layer.expand(new_degree * 2)
        else:
            self.fourier_degree = new_degree
            self.fourier_layer = FourierLayer(new_degree)
            # Expand linear layer to match new Fourier features
            self.linear_layer.reduce(new_degree * 2)


def train_model(model, optimizer, criterion, train_loader, val_loader, num_epochs, min_degree, max_degree, patience=10):
    min_diff = 0.001
    lowest_loss = torch.inf
    lowest_loss_degree = 0
    epoch = 0
    lowest_loss_per_degree_change = 0
    epochs_without_improvement = 0  # Counter for early stopping

    while epoch < num_epochs:
        model.train()
        train_loss = 0.0
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            # print(outputs.dtype, targets.dtype)
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

        print(
            f"Epoch {(epoch + 1):4}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Lowest Loss: {lowest_loss:.4f} ({lowest_loss_degree}), Fourier Degree: {model.fourier_degree}")

        # Dynamic adjustment of fourier_degree
        current_degree = model.fourier_degree
        if (epoch + 1) % 100 == 0:
            if abs(lowest_loss - train_loss) < min_diff:
                change_lowest_loss_per_degree_change = False
                if lowest_loss_per_degree_change > lowest_loss:
                    if model.fourier_degree + 1 < max_degree:
                        change_lowest_loss_per_degree_change = True
                        model.update_fourier_degree(model.fourier_degree + 1)
                elif model.fourier_degree - 1 > min_degree:
                    change_lowest_loss_per_degree_change = True
                    model.update_fourier_degree(model.fourier_degree - 1)

                optimizer = optim.Adam(model.parameters(), lr=0.01)

                if change_lowest_loss_per_degree_change:
                    lowest_loss_per_degree_change = lowest_loss
                    if train_loss < lowest_loss_per_degree_change:
                        lowest_loss_per_degree_change = train_loss
                        lowest_loss = train_loss
            elif train_loss > 0.01:
                if model.fourier_degree + 1 < max_degree:
                    change_lowest_loss_per_degree_change = True
                    model.update_fourier_degree(model.fourier_degree + 1)
                    optimizer = optim.Adam(model.parameters(), lr=0.01)

        if train_loss < lowest_loss:
            lowest_loss = train_loss
            lowest_loss_degree = current_degree
            epochs_without_improvement = 0  # Reset counter if we improve
        elif not train_loss > 1:
            epochs_without_improvement += 1  # Increment counter if no improvement

        # Check for early stopping
        if epochs_without_improvement >= patience:
            print(f"Early stopping triggered after {epoch + 1} epochs.")
            break

        epoch += 1


class DualOutput:
    def __init__(self, filename):
        self.file = open(filename, 'a')  # Open the file in append mode

    def write(self, message):
        # Write the message to both the file and stdout
        self.file.write(message)
        # Use the original stdout to print to console
        sys.__stdout__.write(message)

    def flush(self):
        # Flush the file buffer
        self.file.flush()

    def close(self):
        # Close the file when done
        self.file.close()


# Example usage
if __name__ == "__main__":
    timestamp = datetime.now()
    dual_output = DualOutput(
        f'tests/tmp/{timestamp.strftime("%Y-%m-%d-%H:%M")}_log.txt')
    sys.stdout = dual_output

    samplerate = 44100
    function = 'nice'

    # Create a simple dataset
    inputs = torch.linspace(-torch.pi, torch.pi, 250).unsqueeze(1)
    targets = torch.tensor(
        [predefined_functions.predefined_functions_dict[function](x) for x in inputs], dtype=torch.float32).unsqueeze(1)
    # print(targets.shape)

    dataset = torch.utils.data.TensorDataset(inputs, targets)
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=100)
    val_loader = torch.utils.data.DataLoader(
        dataset, batch_size=100)  # For simplicity, using the same data

    initial_degree = 100
    model = SimpleModel(initial_degree)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()

    # Train the model with dynamic degree adjustment
    train_model(model, optimizer, criterion, train_loader,
                val_loader, num_epochs=1000, min_degree=1, max_degree=300)

    t2 = 2 * np.pi * np.linspace(0, 1, samplerate)
    t2_tensor = torch.tensor(t2, dtype=torch.float32).unsqueeze(1)
    with torch.no_grad():
        plt.plot(t2, model(t2_tensor).cpu().numpy(), label=f"model")
        plt.plot(
            t2, [predefined_functions.predefined_functions_dict[function](x) for x in (t2-np.pi)], label="actual")

    plt.xlabel('x (radians)')
    plt.ylabel('sin(x)')
    plt.title('Sine Wave')
    plt.grid(True)
    plt.legend()
    plt.show()

    sys.stdout.close()  # Close the file
    sys.stdout = sys.__stdout__  # Restore original stdout
