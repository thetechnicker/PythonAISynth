from matplotlib.animation import FuncAnimation
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
        self.frequencies_sin = nn.Parameter(
            torch.arange(1, degree+1, 1, dtype=torch.float32),
            # requires_grad=False
        )
        self.frequencies_cos = nn.Parameter(
            torch.arange(1, degree+1, 1, dtype=torch.float32),
            # requires_grad=False
        )
        self.a1 = nn.Parameter(torch.ones((degree)))
        self.a2 = nn.Parameter(torch.ones((degree)))
        self.c1 = nn.Parameter(torch.zeros((degree)))
        self.c2 = nn.Parameter(torch.zeros((degree)))
        self.d = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        y1 = self.a1 * torch.sin(self.frequencies_sin * x+self.c1)
        y2 = self.a2 * torch.cos(self.frequencies_cos * x+self.c2)
        z = torch.sum(y1, dim=-1)+torch.sum(y2, dim=-1)+self.d
        return z, y1, y2


def regresion(x, y, degree):
    # Convert to PyTorch tensors
    x_tensor = torch.FloatTensor(x).view(-1, 1)
    y_tensor = torch.FloatTensor(y).view(-1, 1)

    # Instantiate the model
    model = FourierRegresionModel(degree)

    for name, param in model.named_parameters():
        print(
            f"Layer: {name} | Size: {param.size()}")  # | Values:\n{param}\n------------------------------")

    # Step 3: Define loss function and optimizer
    criterion = nn.HuberLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    dataset = TensorDataset(x_tensor, y_tensor)
    dataloader = DataLoader(dataset, batch_size=50)  # , shuffle=True)

    # yield y, np.zeros((*x.shape, 5)), np.zeros((*x.shape, 5))

    # Step 4: Train the model
    num_epochs = 10000
    try:
        for epoch in range(num_epochs):
            model.train()

            # Forward pass
            for batch_x, batch_y in dataloader:
                outputs, _, _ = model(batch_x)
                loss = criterion(outputs, batch_y)

                optimizer.zero_grad()
                # Backward pass and optimization
                loss.backward()
                optimizer.step()
            if (epoch+1) % 10 == 0:
                print(
                    f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.10f}')

            model.eval()
            with torch.no_grad():
                predicted, y1, y2 = model(x_tensor)
            yield predicted.numpy(), y1.numpy(), y2.numpy()
    except KeyboardInterrupt:
        print()

    for name, param in model.named_parameters():
        print(
            f"Layer: {name} | Size: {param.size()} | Values:\n{param}\n------------------------------")

    # Step 5: Visualize the results
    model.eval()
    with torch.no_grad():
        predicted, y1, y2 = model(x_tensor)

    yield predicted.numpy(), y1.numpy(), y2.numpy()


def main():
    # Generate synthetic data
    x = np.linspace(-2*np.pi, 2*np.pi, 1000)
    y = np.sin(x+np.random.randn()) +\
        1/2*np.sin(2*x+np.random.randn()) +\
        1/3*np.sin(3*x-np.random.randn()) + \
        np.cos(4*x-np.random.randn())

    degree = 10

    regresion_gen = regresion(x, y, degree)

    fig, ax = plt.subplots()
    ax.scatter(x, y, label='Original Data', color='blue')

    line1, = ax.plot(x, np.zeros_like(x), label='Fitted Line', color='black')
    line2 = ax.plot(x, np.zeros((*x.shape, degree)), color='red')
    line3 = ax.plot(x, np.zeros((*x.shape, degree)), color='green')

    ax.axhline(y=np.mean(y), color='green',
               linestyle='--', label='Vertical Shift (c)')
    ax.set_title('Simple Regression with Cosine Function')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.grid(True)
    ax.legend()

    def update(data):
        line1.set_ydata(data[0])
        for i, line in enumerate(line2):
            line.set_ydata(data[1][:, i])
        for i, line in enumerate(line3):
            line.set_ydata(data[2][:, i])
        return line1, *line2, *line3

    ani = FuncAnimation(fig,
                        update,
                        frames=regresion_gen,
                        interval=10,
                        blit=True,
                        repeat=False)
    plt.show()


if __name__ == "__main__":
    main()
