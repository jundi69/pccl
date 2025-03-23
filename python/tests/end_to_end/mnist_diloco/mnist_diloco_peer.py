import os

import zlib
from time import sleep
from typing import List, Optional

import pccl
from pccl import Communicator, SharedState, TensorInfo, ReduceOp, Attribute, PCCLError, ReduceOperandDescriptor, \
    DataType, DistributionHint, QuantizationOptions, QuantizationAlgorithm
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
learning_rate = 3e-4
IS_CI = os.getenv('IS_CI', '0') == '1'
max_steps = 1024
NUM_INNER_STEPS = 16

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

def log_info(msg: str):
    print(msg)


RANK: int = int(os.getenv('RANK', "0"))


def all_reduce_multiple_with_retry(communicator: Communicator, tensors: List[torch.Tensor], op: ReduceOp, it: int = 0):
    world_size = communicator.get_attribute(Attribute.GLOBAL_WORLD_SIZE)

    def launch_all_reduce(x: torch.Tensor, tag: int):
        op_desc = ReduceOperandDescriptor(
            datatype=DataType.FLOAT,
            distribution_hint=DistributionHint.NORMAL
        )
        quant_desc = QuantizationOptions(
            quantized_datatype=DataType.UINT8,
            algorithm=QuantizationAlgorithm.MIN_MAX
        )
        return communicator.all_reduce_async(x, x, operand_descriptor=op_desc,
                                               quantization_options=quant_desc,
                                               op=op, tag=tag)

    handles: List[Optional[pccl.AsyncReduceHandle]] = [None for _ in range(len(tensors))]
    done_handles = set()

    for tensor_index in range(len(tensors)):
        handles[tensor_index] = launch_all_reduce(tensors[tensor_index], tensor_index)

    while world_size > 1:
        all_done = True
        for tensor_index in range(len(tensors)):
            handle = handles[tensor_index]
            if handle is None:
                if tensor_index in done_handles:
                    continue
                else:
                    handle = handles[tensor_index] = launch_all_reduce(tensors[tensor_index], tensor_index)

            log_debug(f"(RANK={RANK}, it={it}) all_reduce_async wait({tensor_index})")
            is_success, status, info = handle.wait()
            world_size = communicator.get_attribute(Attribute.GLOBAL_WORLD_SIZE)
            if not is_success:
                log_debug(f"(RANK={RANK}, it={it}) all_reduce_async({tensor_index}) failed; New world_size: {world_size}")
                log_info(f"(RANK={RANK}, it={it}) waiting for all async operations to complete before retrying...")
                for j in range(len(tensors)):
                    if j == tensor_index:
                        continue
                    handle = handles[j]
                    if handle is not None:
                        is_success, status, info = handle.wait()
                        if is_success:
                            done_handles.add(j)
                        handles[tensor_index] = None
                log_info(f"(RANK={RANK}, it={it}) all async operations awaited; retrying...")
                handles[tensor_index] = None
                all_done = False
                break
            assert info is not None
            log_debug(
                f"(RANK={RANK}, it={it}) Reduce completed RX: {info.rx_bytes}, TX: {info.tx_bytes}; world_size: {info.world_size}")
            handles[tensor_index] = None
            done_handles.add(tensor_index)

        if all_done:
            break

    if world_size == 1:
        # await all handles
        for handle in handles:
            if handle is not None:
                handle.wait()
        return False

    return True

def print_outer_crc32s(outer_optimizer: optim.Optimizer, it: int):
    for outer_enry in outer_optimizer.state.values():
        for key, value in outer_enry.items():
            if isinstance(value, torch.Tensor):
                checksum = compute_crc32(value)
                log_debug(f"(RANK={RANK}, it={it}) {key} checksum: {checksum}")


def update_topology_with_retries(communicator: Communicator, it: int = 0):
    log_debug(f"(RANK={RANK}, it={it}) pccl::update_topology()")
    while True:
        try:
            communicator.update_topology()
            break
        except PCCLError as e:
            log_debug(f"(RANK={RANK}, it={it}) pccl::update_topology() failed: {e}; retrying...")
            continue


def main():
    model = NeuralNet(input_size, hidden_sizes, num_classes).to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    inner_optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # communicator
    communicator = Communicator('127.0.0.1:48148', 0)
    communicator.connect(n_attempts=15)
    log_info(f"(RANK={RANK}) Connected to the master node; PID={os.getpid()}")

    # Inner optimizer state is not part of the shared state for DILoCo
    outer_params = {
        param_name: torch.nn.Parameter(param.detach().clone()) for param_name, param in model.named_parameters()
    }

    # Ensure order of values list matches iterator order in named_parameters
    outer_params_values = []
    for param_name, param in model.named_parameters():
        outer_params_values.append(outer_params[param_name])

    outer_optimizer = optim.SGD(outer_params_values, lr=learning_rate, momentum=0.9)

    # perform a dummy forward pass to initialize the outer optimizer state
    for outer_p in outer_params_values:
        outer_p.grad = torch.zeros_like(outer_p, device=device)
    outer_optimizer.step()
    log_debug(f"(RANK={RANK}) Initialized optimizer state")

    shared_state_dict = {}
    for name, param in model.named_parameters():
        shared_state_dict[name] = param

    for name, outer_p in outer_params.items():
        # Reference outer optimizer state in shared state
        state = outer_optimizer.state[outer_p]
        momentum_buffer = state["momentum_buffer"]
        shared_state_dict[f"{name}_momentum_buffer"] = momentum_buffer

    entries = [TensorInfo.from_torch(tensor, name, allow_content_inequality=False)
               for name, tensor in shared_state_dict.items()]
    shared_state = SharedState(entries)
    log_info(f"(RANK={RANK}) Initialized shared state")

    train_it = enumerate(train_loader)
    try:
        it = 0
        num_syncs = 0
        while True:
            it += 1
            if it > 1:
                update_topology_with_retries(communicator, it)

            world_size = communicator.get_attribute(Attribute.GLOBAL_WORLD_SIZE)

            if world_size < 2:
                sleep(1)
                continue

            # synchronize shared state
            sync_info = communicator.sync_shared_state(shared_state)
            num_syncs += 1

            assert sync_info is not None
            if num_syncs > 1:
                assert sync_info.rx_bytes == 0  # hard assert no divergence between peers in shared state beyond the initial transfer when a peer joins

            # initialize outer state on first sync
            if num_syncs == 1:
                with torch.no_grad():
                    for outer_param, inner_param in zip(outer_params_values, model.parameters()):
                        outer_param.copy_(inner_param.data)

            # print crc32 hashes of outer optimizer state
            print_outer_crc32s(outer_optimizer, it)

            log_info(f"(RANK={RANK}, it={it}) Performing inner steps...")
            for j in range(NUM_INNER_STEPS):
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
                inner_optimizer.zero_grad(set_to_none=False)
                loss.backward()

                # Perform local step
                inner_optimizer.step()

                log_debug(f"(RANK={RANK}, it={it}) loss: {loss.item()}, revision: {shared_state.revision}")

            log_info(f"(RANK={RANK}, it={it}) Performing outer step...")

            # perform outer step
            if outer_params is not None:
                for outer_param, param in zip(outer_params_values, model.parameters()):
                    outer_param.grad = outer_param - param.data

                if not all_reduce_multiple_with_retry(communicator, [outer_param.grad for outer_param in outer_params_values], ReduceOp.AVG, it):
                    # We just play DiLoCo with ourselves... it's the only thing we can do here really
                    pass

            outer_optimizer.step()
            outer_optimizer.zero_grad(set_to_none=False)

            # print crc32 hashes of outer optimizer state
            print_outer_crc32s(outer_optimizer, it)

            # copy outer state into inner state
            with torch.no_grad():
                for inner_param, outer_param in zip(model.parameters(), outer_params_values):
                    inner_param.copy_(outer_param)

            shared_state.revision += 1


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
