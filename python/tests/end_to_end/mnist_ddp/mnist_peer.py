import os
import zlib
from time import sleep
from typing import List

import pccl
from pccl import Communicator, SharedState, TensorInfo, ReduceOp, Attribute, PCCLError, ReduceOperandDescriptor, \
    DataType, DistributionHint, QuantizationOptions
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


def compute_crc32(tensor: torch.Tensor) -> int:
    tensor_cpu = tensor.detach().cpu()
    tensor_contiguous = tensor_cpu.contiguous()
    tensor_np = tensor_contiguous.numpy()
    tensor_bytes = tensor_np.tobytes()
    checksum = zlib.crc32(tensor_bytes)
    return checksum


USE_CUDA = os.getenv("MNIST_USE_CUDA", None) == "1"

# Set device
device = torch.device("cpu") if not USE_CUDA else torch.device("cuda")

# Define hyperparameters
input_size = 28 * 28  # MNIST images are 28x28
# hidden_sizes = [1024, 4096, 4096, 4096, 4096, 1024]
hidden_sizes = [128]
num_classes = 10  # Digits 0-9
batch_size = 32
learning_rate = 0.001
IS_CI = os.getenv('IS_CI', '0') == '1'
max_steps = int(os.getenv("MAX_STEPS", "256"))

# MNIST dataset (images and labels)
train_dataset = datasets.MNIST(root='./data', train=True, transform=transforms.ToTensor(), download=True)
test_dataset = datasets.MNIST(root='./data', train=False, transform=transforms.ToTensor(), download=True)

# Data loader (input pipeline)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

num_threads = os.getenv('USE_TORCH_NUM_THREADS', None)
if num_threads is not None:
    torch.set_num_threads(int(num_threads))
else:
    torch.set_num_threads(1)

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


LOG_DEBUG = False if IS_CI else True


def log_debug(msg: str):
    if LOG_DEBUG:
        print(msg)


_IS_FIRST_MSG = True


def log_info(msg: str):
    global _IS_FIRST_MSG
    if _IS_FIRST_MSG:
        event_name = os.getenv('FIRST_MSG_EVENT_NAME', None)
        if event_name is not None:
            import win32event
            event_handle = win32event.OpenEvent(win32event.EVENT_ALL_ACCESS, False, event_name)
            if event_handle:
                win32event.SetEvent(event_handle)

        _IS_FIRST_MSG = False
    print(msg)


HOST: str = '127.0.0.1:48148'
RANK: int = int(os.getenv('RANK', "0"))


def generate_random_vectors_64(d: int,
                               device=None,
                               dtype=torch.float32):
    """
    Generate 64 random vectors each of dimension d.
    Shape: (64, d).
    """
    gen = torch.Generator()
    gen.manual_seed(42)
    return torch.randn((64, d), device=device, dtype=dtype, generator=gen)


def random_projection_64(weights: torch.Tensor) -> torch.Tensor:
    """
    Project 'weights' (shape [d]) onto the 64 random vectors (shape [64, d]).

    Returns a tensor of shape (64,) which is the 64D random projection.
    """
    return generate_random_vectors_64(len(weights)) @ weights


def main():
    model = NeuralNet(input_size, hidden_sizes, num_classes).to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # communicator
    communicator: Communicator = Communicator(HOST, 0)
    communicator.connect(n_attempts=15)
    log_info(f"(RANK={RANK}) Connected to the master node; PID={os.getpid()}")

    # perform a dummy forward pass to initialize the optimizer state
    for p in model.parameters():
        p.grad = torch.zeros_like(p, device=device)  # set all gradients to zero
    optimizer.step()
    print(f"(RANK={RANK}) Initialized optimizer state")

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
    print(f"(RANK={RANK}) Initialized shared state")

    # Training loop
    train_it = enumerate(train_loader)
    try:
        it = 0
        num_syncs = 0
        while True:
            it += 1
            if it > 1:
                log_debug(f"(RANK={RANK}, it={it}) update_topology()")
                while True:
                    try:
                        communicator.update_topology()
                        break
                    except PCCLError as e:
                        log_debug(f"(RANK={RANK}, it={it}) update_topology() failed: {e}; retrying...")
                        continue

            world_size = communicator.get_attribute(Attribute.CURRENT_WORLD_SIZE)

            if world_size < 2:
                sleep(1)
                continue

            params = torch.cat([p.view(-1) for p in model.parameters()])
            # log_debug(f"(RANK={RANK}, it={i}, ws={world_size}) [pre shared state sync] params crc32 hash: {compute_crc32(params)}, lindenstrauss: {random_projection_64(params)}")
            log_debug(
                f"(RANK={RANK}, it={it}, ws={world_size}) [pre shared state sync] params crc32 hash: {compute_crc32(params)}")

            sync_info = communicator.sync_shared_state(shared_state)
            num_syncs += 1
            log_debug(
                f"(RANK={RANK}, it={it}) shared_state.revision: {shared_state.revision}, sync_info (tx_bytes={sync_info.tx_bytes}, rx_bytes={sync_info.rx_bytes})")
            if shared_state.revision >= max_steps:
                twz = os.getenv("DONT_EXIT_BEFORE_REACHED_WORLD_SIZE")
                if twz is not None:
                    if int(twz) > world_size:
                        continue

                log_debug(f"(RANK={RANK}, it={it}) Training completed")
                break

            # collect model parameters in one contiguous tensor
            params = torch.cat([p.view(-1) for p in model.parameters()])
            log_debug(
                f"(RANK={RANK}, it={it}) [post shared state sync] params crc32 hash: {compute_crc32(params)}")
            # log_debug(f"(RANK={RANK}, it={i}) [post shared state sync] params crc32 hash: {compute_crc32(params)}, lindenstrauss: {random_projection_64(params)}")

            assert sync_info is not None
            if num_syncs > 1:
                assert sync_info.rx_bytes == 0

            try:
                batch_idx, (images, labels) = next(train_it)
            except StopIteration:
                log_debug(f"(RANK={RANK}, it={it}) End of epoch")
                train_it = enumerate(train_loader)
                batch_idx, (images, labels) = next(train_it)

            images, labels = images.to(device), labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            optimizer.zero_grad(set_to_none=False)
            loss.backward()

            # collect gradients in one contiguous tensor
            grads = torch.cat([p.grad.view(-1) for p in model.parameters() if p.grad is not None]).to('cpu')

            while world_size > 1:
                log_debug(f"(RANK={RANK}, it={it}) all_reduce_async()")
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
                world_size = communicator.get_attribute(Attribute.CURRENT_WORLD_SIZE)
                if not is_success:
                    log_debug(f"(RANK={RANK}, it={it}) all_reduce_async() failed: {status}; retrying...")
                    continue
                assert info is not None
                log_debug(
                    f"(RANK={RANK}, it={it}) Reduce completed RX: {info.rx_bytes}, TX: {info.tx_bytes}; world_size: {info.world_size}")
                break

            if world_size == 1:
                # drop current step, as we are alone in the run and whatever we just computed would induce too much noise if we stepped here.
                # If one accepts the pattern that one waits until the world size is at least two, it would be erroneous to step here.
                print(
                    "All peers have left except this peer. Dropping current step to avoid inducing too much variance with our local batch!")
                continue

            # print hash of the gradients tensor content
            # log_debug(f"(RANK={RANK}, it={it}) grads hash: {compute_crc32(grads)}")

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

            log_debug(f"(RANK={RANK}, it={it}) loss: {loss.item()}, revision: {shared_state.revision}")

            if shared_state.revision % 5 == 0:
                log_info(f"(RANK={RANK}, it={it}) loss: {loss.item()}, revision: {shared_state.revision}")

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

        log_info(f"Test Accuracy: {100 * correct / total:.2f}%")

    # Save the model
    torch.save(model.state_dict(), "mnist_model.pth")
    log_info("Model saved to mnist_model.pth")


if __name__ == '__main__':
    main()
    exit(0)
