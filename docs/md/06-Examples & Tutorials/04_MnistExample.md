# MNIST Example (Python)
```python
import os
import time
from typing import List

import pccl
from pccl import Communicator, SharedState, TensorInfo, ReduceOp, PCCLError, ReduceOperandDescriptor, \
    DataType, DistributionHint, QuantizationOptions
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Set device
device = torch.device("cpu")

# Define hyperparameters
input_size = 28 * 28  # MNIST images are 28x28
# hidden_sizes = [1024, 4096, 4096, 4096, 4096, 1024]
hidden_sizes = [256]
num_classes = 10  # Digits 0-9
batch_size = 128
learning_rate = 0.001
IS_CI = os.getenv('IS_CI', '0') == '1'
max_steps = IS_CI and 512 or 2048

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


MASTER_HOST = "127.0.0.1"
MASTER_PORT = 48148

def main():
    model = NeuralNet(input_size, hidden_sizes, num_classes).to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # communicator
    communicator: Communicator = Communicator(f"{MASTER_HOST}:{MASTER_PORT}", 0)
    communicator.connect(n_attempts=15)
    print(f"Connected to the master node; PID={os.getpid()}")

    # perform a dummy forward pass to initialize the optimizer state
    for p in model.parameters():
        p.grad = torch.zeros_like(p)  # set all gradients to zero
    optimizer.step()
    print(f"Initialized optimizer state")

    # Reference model and optimizer state from shared state struct
    shared_state_dict = {}
    for name, param in model.named_parameters():
        shared_state_dict[name] = param

        # Access optimizer state
        state = optimizer.state[param]
        exp_avg = state.get('exp_avg')
        exp_avg_sq = state.get('exp_avg_sq')
        step_tensor = state.get('step')

        if exp_avg is None or exp_avg_sq is None or step_tensor is None:
            raise ValueError(f"Optimizer state for parameter '{name}' is not initialized.")

        # Add optimizer state tensors with associated names
        shared_state_dict[f"{name}_m1"] = exp_avg
        shared_state_dict[f"{name}_m2"] = exp_avg_sq
        shared_state_dict[f"{name}_step"] = step_tensor

    entries = [TensorInfo.from_torch(tensor, name, allow_content_inequality=False) for name, tensor in
               shared_state_dict.items()]
    shared_state: SharedState = SharedState(entries)
    print(f"Initialized shared state")

    # Training loop
    train_it = enumerate(train_loader)
    try:
        it = 0
        num_syncs = 0
        while True:
            it += 1
            if it > 1:
                print(f"(it={it}) update_topology()")
                while True:
                    try:
                        communicator.update_topology()
                        break
                    except PCCLError as e:
                        print(f"(it={it}) update_topology() failed: {e}; retrying...")
                        continue

            world_size = communicator.get_attribute(pccl.Attribute.GLOBAL_WORLD_SIZE)

            if world_size > 1:
                while True:
                    try:
                        communicator.optimize_topology()
                        break
                    except pccl.PCCLError as e:
                        print(f"(it={it}) OptimizeTopology failed => {e}. Retrying...")
                        time.sleep(0.1)
                world_size = communicator.get_attribute(pccl.Attribute.GLOBAL_WORLD_SIZE)
    

            if world_size < 2:
                # alone => no ring-based operation => wait
                print("[Peer] alone => sleeping.")
                time.sleep(1)
                continue

            # Perform cuda device synchronization
            # if your shared state partially or fully resides on the GPU we must wait until all currently dispatched kernels have completed
            # to avoid validating or potentially transmitting data that is currently being in-place modified.
            torch.cuda.synchronize(device)
            
            sync_info = communicator.sync_shared_state(shared_state)
            num_syncs += 1
            if shared_state.revision >= max_steps:
                print(f"(it={it}) Training completed")
                break

            assert sync_info is not None
            if num_syncs > 1:
                assert sync_info.rx_bytes == 0

            try:
                batch_idx, (images, labels) = next(train_it)
            except StopIteration:
                print(f"(it={it}) End of epoch")
                train_it = enumerate(train_loader)
                batch_idx, (images, labels) = next(train_it)

            images, labels = images.to(device), labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            # set_to_none=False is required, otherwise shared state references will become invalid!
            optimizer.zero_grad(set_to_none=False)
            loss.backward()

            # collect gradients in one contiguous tensor
            grads = torch.cat([p.grad.view(-1) for p in model.parameters() if p.grad is not None])

            while world_size > 1:
                print(f"(it={it}) all_reduce_async()")
                op_desc = ReduceOperandDescriptor(
                    datatype=DataType.FLOAT,
                    distribution_hint=DistributionHint.NORMAL
                )
                quant_desc = QuantizationOptions(
                    quantized_datatype=DataType.FLOAT,
                    algorithm=pccl.QuantizationAlgorithm.NONE
                )
                handle = communicator.all_reduce_async(grads, grads, operand_descriptor=op_desc,
                                                       quantization_options=quant_desc, op=ReduceOp.SUM)
                is_success, status, info = handle.wait()
                world_size = communicator.get_attribute(pccl.Attribute.GLOBAL_WORLD_SIZE)
                if not is_success:
                    print(f"(it={it}) all_reduce_async() failed: {status}; retrying...")
                    continue
                assert info is not None
                print(
                    f"(it={it}) Reduce completed RX: {info.rx_bytes}, TX: {info.tx_bytes}; world_size: {info.world_size}")
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

            if shared_state.revision % 5 == 0:
                print(f"(it={it}) loss: {loss.item()}, revision: {shared_state.revision}")

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
```