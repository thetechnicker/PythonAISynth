from matplotlib.ticker import MaxNLocator
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset


class FourierRegresionModel(nn.Module):
    def __init__(self, degree):
        super(FourierRegresionModel, self).__init__()
        # Define three trainable parameters
        # torch.tensor([[1, 1/2, 0]])
        self.a = nn.Parameter(torch.ones((1, degree)))  # Parameter a
        # self.b = nn.Parameter(torch.randn(1, degree))  # Parameter b
        self.b = torch.arange(1, degree+1, 1)  # .reshape((1, degree))
        self.c = nn.Parameter(torch.zeros((1, degree)))  # Parameter c

    def forward(self, x):
        y = self.a * torch.sin(self.b * x)  # + self.c
        return torch.sum(y, dim=-1)


def regresion():
    # Generate synthetic data
    x = np.linspace(0, 2*np.pi, 100)
    y = np.sin(x) + 1/2*np.sin(2*x) + 1/3*np.sin(3*x)

    # Convert to PyTorch tensors
    x_tensor = torch.FloatTensor(x).view(-1, 1).unsqueeze(1)
    y_tensor = torch.FloatTensor(y).view(-1, 1).unsqueeze(1)

    # Instantiate the model
    model = FourierRegresionModel(5)

    for name, param in model.named_parameters():
        print(
            f"Layer: {name} | Size: {param.size()} | Values:\n{param}\n------------------------------")

    print(model.b)

    # Step 3: Define loss function and optimizer
    criterion = nn.HuberLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    dataset = TensorDataset(x_tensor, y_tensor)
    dataloader = DataLoader(dataset, batch_size=10, shuffle=False)

    # Step 4: Train the model
    num_epochs = 1000
    try:
        for epoch in range(num_epochs):
            model.train()

            # Forward pass
            for batch_x, batch_y in dataloader:
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)

                optimizer.zero_grad()
                # Backward pass and optimization
                loss.backward()
                optimizer.step()
            if (epoch+1) % 10 == 0:
                print(
                    f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.10f}')
    except KeyboardInterrupt:
        print()

    for name, param in model.named_parameters():
        print(
            f"Layer: {name} | Size: {param.size()} | Values:\n{param}\n------------------------------")

    print(model.b)

    # Step 5: Visualize the results
    model.eval()
    with torch.no_grad():
        predicted = model(x_tensor).numpy()
        print(predicted.shape)

    return x, y, predicted


def main():
    x, y, predicted = regresion()

    plt.scatter(x, y, label='Original Data', color='blue')

    if predicted is not None:
        plt.plot(x, predicted, label='Fitted Line', color='red')

    plt.title('Simple Regression with Cosine Function')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.axhline(y=np.mean(y), color='green',
                linestyle='--', label='Vertical Shift (c)')
    plt.grid(True)
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
