import os
import signal
from time import sleep
from typing import List

import pccl
from pccl import Communicator, SharedState, TensorInfo, ReduceInfo, ReduceOp, Attribute, PCCLError
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


def human_readable_bytes(size: int) -> str:
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size < 1024.0:
            break
        size /= 1024.0
    return f"{size:.2f} {unit}"


# Set device
device = torch.device("cpu")

# Define hyperparameters
input_size = 28 * 28  # MNIST images are 28x28
# hidden_sizes = [1024, 4096, 4096, 4096, 4096, 1024]
hidden_sizes = [1024, 1024]
num_classes = 10  # Digits 0-9
batch_size = 64
learning_rate = 0.001
max_steps = 10000

# MNIST dataset (images and labels)
train_dataset = datasets.MNIST(root='./data', train=True, transform=transforms.ToTensor(), download=True)
test_dataset = datasets.MNIST(root='./data', train=False, transform=transforms.ToTensor(), download=True)

# Data loader (input pipeline)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)


# Define a simple neural network
class NeuralNet(nn.Module):
    def __init__(self, input_size: int, hidden_sizes: List[int], num_classes: int):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_sizes[0])
        self.relu = nn.ReLU()
        self.fcs = nn.ModuleList(
            [nn.Linear(hidden_sizes[i], hidden_sizes[i + 1]) for i in range(len(hidden_sizes) - 1)])
        self.fc2 = nn.Linear(hidden_sizes[-1], num_classes)

    def forward(self, x):
        x = x.view(-1, input_size)  # Flatten the image
        x = self.fc1(x)
        x = self.relu(x)
        for fc in self.fcs:
            x = fc(x)
            x = self.relu(x)
        x = self.fc2(x)
        return x


HOST: str = '127.0.0.1:48148'
RANK: int = int(os.getenv('RANK', "0"))


def main():
    model = NeuralNet(input_size, hidden_sizes, num_classes).to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # communicator
    communicator: Communicator = Communicator(HOST, 0)
    n_attempts = 5
    for attempt in range(n_attempts):
        try:
            communicator.connect()
            break
        except PCCLError as e:
            print(
                f"(RANK={RANK}) Failed to connect to the master node: {e}; (Attempt {attempt + 1}/{n_attempts})")

            sleep(1)
    else:
        assert False, f"(RANK={RANK}) Failed to connect to the master node"
    print(f"(RANK={RANK}) Connected to the master node")

    # poll shared state for model parameters
    shared_state_dict = {}
    for name, param in model.named_parameters():
        shared_state_dict[name] = param

    shared_state: SharedState = SharedState([
        TensorInfo.from_torch(param, name) for name, param in shared_state_dict.items()
    ])

    # Training loop
    try:
        i = 0
        while shared_state.revision < max_steps:
            i += 1
            if i > 1:
                print(f"(RANK={RANK}, it={i}) update_topology()")
                communicator.update_topology()
            print(f"(RANK={RANK}, it={i}) sync_shared_state()")
            communicator.sync_shared_state(shared_state)
            world_size = communicator.get_attribute(Attribute.CURRENT_WORLD_SIZE)

            if world_size < 2:
                sleep(0.1)
                continue

            for batch_idx, (images, labels) in enumerate(train_loader):
                images, labels = images.to(device), labels.to(device)

                # Forward pass
                outputs = model(images)
                loss = criterion(outputs, labels)

                # Backward pass and optimization
                optimizer.zero_grad(set_to_none=False)
                loss.backward()

                # collect gradients in one contiguous tensor
                grads = torch.cat([p.grad.view(-1) for p in model.parameters() if p.grad is not None])

                while True:
                    print(f"(RANK={RANK}, it={i}) all_reduce_async()")
                    handle = communicator.all_reduce_async(grads, grads, numel=grads.numel(), op=ReduceOp.SUM)
                    is_success, status, info = handle.wait()
                    assert is_success == True, f"All reduce failed with stats: {status}"
                    assert info is not None
                    print(f"((RANK={RANK}, it={i}) Reduce completed RX: {info.rx_bytes}, TX: {info.tx_bytes}")
                    break

                # scatter gradients back to model parameters
                offset = 0
                for p in model.parameters():
                    if p.grad is None:
                        continue
                    numel = p.numel()
                    p.grad.data.copy_(grads[offset:offset + numel].view_as(p.grad))
                    offset += numel

                # Update parameters
                optimizer.step()
                shared_state.revision += 1
                if shared_state.revision >= max_steps:
                    break

                print(f"(RANK={RANK}, it={i}) loss: {loss.item()}, revision: {shared_state.revision}")
    except Exception as e:
        print(f"Training aborted with exception: {e}")
        exit(1)

    # Test the model
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        print(f"Test Accuracy: {100 * correct / total:.2f}%")

    # Save the model
    torch.save(model.state_dict(), "mnist_model.pth")
    print("Model saved to mnist_model.pth")


if __name__ == '__main__':
    main()
