"""
This training script can be run both on a single gpu in debug mode,
and also in a larger training run with distributed data parallel (ddp).

Example single GPU:
$ python train_pccl.py --config_path=default_config.json

Example DDP on 4 GPUs on 1 node:
$ torchrun --standalone --nproc_per_node=4 train_pccl.py --config_path=default_config.json

Example DDP on 8 GPUs across 2 nodes:
(Master node, IP 123.456.123.456)
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 --master_addr=123.456.123.456 --master_port=1234 train_pccl.py

(Worker node)
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=1 --master_addr=123.456.123.456 --master_port=1234 train_pccl.py
(If your cluster does not have Infiniband interconnect prepend NCCL_IB_DISABLE=1)

All runtime configuration (hyperparameters, logging settings, etc.) is expected
to come from a JSON file specified by --config_path (default_config.json by default).
"""

import os
import threading
import time
import math
import pickle
import json
import argparse
import pccl
import zlib
from contextlib import nullcontext

import numpy as np
import torch
from pccl import Communicator, Attribute, SharedState, TensorInfo, DataType, ReduceOperandDescriptor, DistributionHint, \
    QuantizationOptions, ReduceOp
from torch.nn.parallel import DistributedDataParallel
from torch.distributed import init_process_group, destroy_process_group

from model import GPTConfig, GPT
from python.examples.nanogptddp.profiler import Profiler, ProfilerCollection


def get_batch(split, config, device_type, device):
    """
    Poor man's data loader. Each call randomly selects `batch_size` samples from
    train or val memmapped .bin files and returns them as x, y tensors.
    """
    data_dir = os.path.join('data', config["dataset"])
    block_size = config["block_size"]
    batch_size = config["batch_size"]

    # We recreate np.memmap every call to avoid memory leaks on repeated usage
    data_path = os.path.join(data_dir, 'train.bin' if split == 'train' else 'val.bin')
    data = np.memmap(data_path, dtype=np.uint16, mode='r')

    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([
        torch.from_numpy((data[i: i + block_size]).astype(np.int64)) for i in ix
    ])
    y = torch.stack([
        torch.from_numpy((data[i + 1: i + 1 + block_size]).astype(np.int64)) for i in ix
    ])

    if device_type == 'cuda':
        x = x.pin_memory().to(device, non_blocking=True)
        y = y.pin_memory().to(device, non_blocking=True)
    else:
        x = x.to(device)
        y = y.to(device)

    return x, y


def compute_crc32(tensor: torch.Tensor) -> int:
    tensor_cpu = tensor.detach().cpu()
    tensor_contiguous = tensor_cpu.contiguous()
    tensor_np = tensor_contiguous.numpy()
    tensor_bytes = tensor_np.tobytes()
    checksum = zlib.crc32(tensor_bytes)
    return checksum


@torch.no_grad()
def estimate_loss(model, ctx, config, get_batch_fn, device_type, device):
    """
    Estimate the mean loss on train and val sets by running `eval_iters` random mini-batches.
    """
    eval_iters = config["eval_iters"]
    out = {}
    model.eval()

    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters, device=device)
        for k in range(eval_iters):
            x, y = get_batch_fn(split, config, device_type, device)
            with ctx:
                _, loss = model(x, y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()  # ensure a Python float
    model.train()
    return out


def get_lr(it, config):
    """
    Learning rate decay scheduler (cosine with warmup). 
    If decay_lr is False, just return the original learning_rate.
    """
    if not config["decay_lr"]:
        return config["learning_rate"]

    learning_rate = config["learning_rate"]
    warmup_iters = config["warmup_iters"]
    lr_decay_iters = config["lr_decay_iters"]
    min_lr = config["min_lr"]

    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * (it + 1) / (warmup_iters + 1)
    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    # clamp the ratio just in case of numerical issues
    decay_ratio = max(0, min(1, decay_ratio))
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (learning_rate - min_lr)


EXPORT_PROFILER_VIDEO = True


def main():
    """
    Main entry point for setting up training, distributed data parallel, loading config,
    initializing model, and running the training loop.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_path",
        type=str,
        default="default_config.json",
        help="Path to JSON configuration file."
    )
    args = parser.parse_args()

    # -------------------------------------------------------------------------
    # 1) Load configuration from JSON
    # -------------------------------------------------------------------------
    with open(args.config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    if (config["dtype"] == "float16") and torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        config["dtype"] = "bfloat16"

    # -------------------------------------------------------------------------
    # 2) Initialize DDP if needed
    # -------------------------------------------------------------------------
    ddp = int(os.environ.get('RANK', -1)) != -1  # check if distributed run
    if ddp:
        init_process_group(backend=config["backend"])
        ddp_rank = int(os.environ['RANK'])
        ddp_local_rank = int(os.environ['LOCAL_RANK'])
        ddp_world_size = int(os.environ['WORLD_SIZE'])
        device = f'cuda:{ddp_local_rank}'
        torch.cuda.set_device(device)
        master_process = (ddp_rank == 0)
        seed_offset = ddp_rank
        # scale down gradient_accumulation_steps according to the world size
        gas = config["gradient_accumulation_steps"]
        assert gas % ddp_world_size == 0, (
            f"gradient_accumulation_steps={gas} must be divisible by WORLD_SIZE={ddp_world_size}"
        )
        config["gradient_accumulation_steps"] = gas // ddp_world_size
    else:
        master_process = config["is_master_process"]
        seed_offset = 0
        ddp_world_size = 1
        device = config["device"]

    # calculate tokens per iteration for reference
    tokens_per_iter = (
            config["gradient_accumulation_steps"] *
            ddp_world_size *
            config["batch_size"] *
            config["block_size"]
    )
    print(f"tokens per iteration will be: {tokens_per_iter:,}")

    # -------------------------------------------------------------------------
    # 3) Setup environment, seeds, output directory
    # -------------------------------------------------------------------------
    if master_process:
        os.makedirs(config["out_dir"], exist_ok=True)
    torch.manual_seed(1337 + seed_offset)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    device_type = 'cuda' if 'cuda' in device else 'cpu'
    ptdtype = {
        'float32': torch.float32,
        'bfloat16': torch.bfloat16,
        'float16': torch.float16
    }[config["dtype"]]
    ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

    # -------------------------------------------------------------------------
    # 4) Prepare data info, possibly read vocab size from meta
    # -------------------------------------------------------------------------
    data_dir = os.path.join('data', config["dataset"])
    meta_path = os.path.join(data_dir, 'meta.pkl')
    meta_vocab_size = None
    if os.path.exists(meta_path):
        with open(meta_path, 'rb') as f:
            meta = pickle.load(f)
        meta_vocab_size = meta['vocab_size']
        print(f"found vocab_size = {meta_vocab_size} (inside {meta_path})")

    # -------------------------------------------------------------------------
    # 5) Create or resume model
    # -------------------------------------------------------------------------
    torch.manual_seed(1337 + seed_offset)
    model_args = dict(
        n_layer=config["n_layer"],
        n_head=config["n_head"],
        n_embd=config["n_embd"],
        block_size=config["block_size"],
        bias=config["bias"],
        vocab_size=None,
        dropout=config["dropout"]
    )

    iter_num = 0
    best_val_loss = 1e9
    init_from = config["init_from"]
    checkpoint = None

    if init_from == 'scratch':
        print("Initializing a new model from scratch.")
        if meta_vocab_size is None:
            print("Defaulting to GPT-2 vocab_size of 50304.")
        model_args['vocab_size'] = meta_vocab_size if meta_vocab_size else 50304
        gptconf = GPTConfig(**model_args)
        model = GPT(gptconf)

    elif init_from == 'resume':
        print(f"Resuming training from {config['out_dir']}")
        ckpt_path = os.path.join(config["out_dir"], 'ckpt.pt')
        checkpoint = torch.load(ckpt_path, map_location=device)
        checkpoint_model_args = checkpoint['model_args']
        # Force certain config attributes to match
        for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
            model_args[k] = checkpoint_model_args[k]
        gptconf = GPTConfig(**model_args)
        model = GPT(gptconf)

        state_dict = checkpoint['model']
        # Some checkpoints might have an unwanted prefix
        unwanted_prefix = '_orig_mod.'
        for k, v in list(state_dict.items()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
        model.load_state_dict(state_dict)

        iter_num = checkpoint['iter_num']
        best_val_loss = checkpoint['best_val_loss']

    elif init_from.startswith('gpt2'):
        print(f"Initializing from OpenAI GPT-2 weights: {init_from}")
        override_args = dict(dropout=config["dropout"])
        model = GPT.from_pretrained(init_from, override_args)
        # read off created config params and store them
        for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
            model_args[k] = getattr(model.config, k)
    else:
        raise ValueError(f"Unknown init_from: {init_from}")

    # Crop down the model block size if needed
    if config["block_size"] < model.config.block_size:
        model.crop_block_size(config["block_size"])
        model_args['block_size'] = config["block_size"]
    model.to(device)

    # -------------------------------------------------------------------------
    # 6) Setup optimizer
    # -------------------------------------------------------------------------
    optimizer = model.configure_optimizers(
        config["weight_decay"],
        config["learning_rate"],
        (config["beta1"], config["beta2"]),
        device_type
    )

    if init_from == 'resume':
        assert checkpoint is not None, "checkpoint must be loaded"
        optimizer.load_state_dict(checkpoint['optimizer'])
        del checkpoint  # free up memory

    if config["compile"]:
        print("Compiling the model... (PyTorch 2.0)")
        model = torch.compile(model)

    if ddp:
        model = DistributedDataParallel(model, device_ids=[int(device.split(":")[-1])])

    # If wandb logging is enabled
    if config["wandb_log"] and master_process:
        import wandb
        wandb.init(
            project=config["wandb_project"],
            name=config["wandb_run_name"],
            config=config
        )

    # Initialize PCCL
    communicator: Communicator = Communicator(config["ccoip_host"], 0)
    communicator.connect(n_attempts=15)
    print("Connected to master.")

    # Build PCCL shared state

    # perform a dummy forward pass to initialize the optimizer state
    for p in model.parameters():
        p.grad = torch.zeros_like(p)  # set all gradients to zero
    optimizer.step()

    shared_state_dict = {}

    # Reference model and optimizer state from shared state struct
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

    # -------------------------------------------------------------------------
    # 7) Training loop
    # -------------------------------------------------------------------------
    raw_model = model.module if ddp else model
    get_batch_fn = get_batch
    local_iter_num = 0
    running_mfu = -1.0

    # Prepare first batch
    x, y = get_batch_fn('train', config, device_type, device)

    world_size: int = communicator.get_attribute(Attribute.GLOBAL_WORLD_SIZE)

    grads_dst = None
    grads = None
    profiler_collection = ProfilerCollection()
    while True:
        t0 = time.time()
        profiler = Profiler()
        with profiler.session("step"):
            if local_iter_num > 0 or world_size == 1:
                with profiler.session("pccl::communicator::update_topology()"):
                    communicator.update_topology()
            world_size = communicator.get_attribute(Attribute.GLOBAL_WORLD_SIZE)

            if world_size < 2:
                print("Waiting for more workers to join...")
                time.sleep(1)
                continue

            if world_size > 1:
                while True:
                    try:
                        communicator.optimize_topology()
                        break
                    except pccl.PCCLError as e:
                        print(f"[Peer] OptimizeTopology failed => {e}. Retrying...")
                        time.sleep(0.1)
                world_size = communicator.get_attribute(pccl.Attribute.GLOBAL_WORLD_SIZE)

            torch.cuda.synchronize(device)

            with profiler.session("pccl::communicator::sync_shared_state()"):
                info = communicator.sync_shared_state(shared_state)
                iter_num = shared_state.revision

            # Determine and set learning rate
            # make sure to use up-to-date shared state revision post sync_shared_state
            # to make sure no learning rate differences between peers occur
            lr = get_lr(iter_num, config)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

            assert info is not None
            print(f"Shared state sync complete: Tx-Bytes: {info.tx_bytes}, Rx-Bytes: {info.rx_bytes}")

            # Evaluate and save checkpoint
            if iter_num % config["eval_interval"] == 0 and master_process:
                with profiler.session("estimate_loss"):
                    losses = estimate_loss(raw_model, ctx, config, get_batch_fn, device_type, device)

                print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

                if config["wandb_log"]:
                    import wandb
                    wandb.log({
                        "iter": iter_num,
                        "train/loss": losses['train'],
                        "val/loss": losses['val'],
                        "lr": lr,
                        "mfu": running_mfu * 100,
                    })

                if losses['val'] < best_val_loss or config["always_save_checkpoint"]:
                    best_val_loss = losses['val']
                    if iter_num > 0:
                        ckpt = {
                            'model': raw_model.state_dict(),
                            'optimizer': optimizer.state_dict(),
                            'model_args': model_args,
                            'iter_num': iter_num,
                            'best_val_loss': best_val_loss,
                            'config': config,
                        }
                        ckpt_path = os.path.join(config["out_dir"], 'ckpt.pt')
                        print(f"Saving checkpoint to {ckpt_path}")
                        with profiler.session("save_checkpoint"):
                            torch.save(ckpt, ckpt_path)

            if iter_num == 0 and config["eval_only"]:
                # If we're only doing evaluation, break right after the first eval
                break

            # Gradient accumulation
            with profiler.session("grad_accum"):
                for micro_step in range(config["gradient_accumulation_steps"]):
                    if ddp:
                        # only sync gradients at last micro step in DDP
                        model.require_backward_grad_sync = (micro_step == config["gradient_accumulation_steps"] - 1)

                    with ctx:
                        with profiler.session("model::forward"):
                            logits, loss = model(x, y)
                        loss = loss / config["gradient_accumulation_steps"]

                    # prefetch next batch while GPU is busy
                    x, y = get_batch_fn('train', config, device_type, device)

                    # backward pass
                    with profiler.session("loss::backward"):
                        loss.backward()

            # Clip gradients BEFORE all reduce to ensure peer synchronization!
            with profiler.session("gradient_clipping"):
                if config["grad_clip"] != 0.0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config["grad_clip"])

            # All reduce gradients with PCCL
            with profiler.session("collect_gradients"):
                if grads is None:
                    grads = torch.cat([p.grad.view(-1) for p in model.parameters() if p.grad is not None]).to(
                        device='cpu')
                else:
                    # scatter grads into existing buffer
                    offset = 0
                    for p in model.parameters():
                        if p.grad is None:
                            continue
                        numel = p.grad.numel()
                        grads[offset:offset + numel].copy_(p.grad.view(-1).cpu())
                        offset += numel

            if grads_dst is None:
                grads_dst = torch.zeros_like(grads)
            else:
                grads_dst.zero_()

            with profiler.session("all_reduce"):
                while world_size > 1:
                    op_desc = ReduceOperandDescriptor(
                        datatype=DataType.FLOAT,
                        distribution_hint=DistributionHint.NORMAL
                    )
                    quant_desc = QuantizationOptions(
                        quantized_datatype=DataType.FLOAT,
                        algorithm=pccl.QuantizationAlgorithm.NONE
                    )

                    start = time.time()
                    with profiler.session("pccl::communicator::all_reduce"):
                        handle = communicator.all_reduce_async(grads, grads_dst, operand_descriptor=op_desc,
                                                               quantization_options=quant_desc, op=ReduceOp.SUM)
                        is_success, status, info = handle.wait()
                        world_size = communicator.get_attribute(Attribute.GLOBAL_WORLD_SIZE)

                    end = time.time()
                    if not is_success:
                        continue  # retry, this can happen e.g. if peers leave
                    assert info is not None
                    total_bytes = info.tx_bytes + info.rx_bytes
                    print(
                        f"step {iter_num}: all reduce completed in {end - start:.2f}s, bandwidth {(total_bytes / 1e6) / (end - start):.2f} MB/s")
                    break
                if world_size == 1:
                    # drop current step, as we are alone in the run and whatever we just computed would induce too much noise if we stepped here.
                    # If one accepts the pattern that one waits until the world size is at least two, it would be erroneous to step here.
                    print("All peers have left except this peer. Dropping current step to avoid inducing too much variance with our local batch!")
                    continue

            # scatter gradients back to model parameters
            with profiler.session("scatter_gradients"):
                offset = 0
                for p in model.parameters():
                    if p.grad is None:
                        continue
                    numel = p.numel()
                    p.grad.data.copy_(grads_dst[offset:offset + numel].view_as(p.grad))
                    offset += numel

            with profiler.session("optimizer::step"):
                optimizer.step()
                with profiler.session("optimizer::zero_grad"):
                    optimizer.zero_grad(set_to_none=False)

            # Logging / timing
            t1 = time.time()
            dt = t1 - t0
            if iter_num % config["log_interval"] == 0 and master_process:
                lossf = loss.item() * config["gradient_accumulation_steps"]  # approximate actual loss
                if local_iter_num >= 5:  # let training loop warm up
                    mfu = raw_model.estimate_mfu(config["batch_size"] * config["gradient_accumulation_steps"], dt)
                    running_mfu = mfu if running_mfu < 0 else 0.9 * running_mfu + 0.1 * mfu
                print(f"iter {iter_num}: loss {lossf:.4f}, time {dt * 1000:.2f}ms, mfu {running_mfu * 100:.2f}%")

            iter_num += 1
            local_iter_num += 1
            shared_state.revision = iter_num

            # Termination
            if iter_num > config["max_iters"]:
                break

        # profiler management
        profiler.print_report()
        if EXPORT_PROFILER_VIDEO:
            profiler_collection.add_profiler(profiler, f"Step {iter_num}")
            if iter_num % 100 == 0:
                profiler_collection.render_as_video(f'timeline_{iter_num}.mp4', fps=15)
                print("Done rendering profiler video")

    # -------------------------------------------------------------------------
    # 8) Cleanup DDP
    # -------------------------------------------------------------------------
    if ddp:
        destroy_process_group()


if __name__ == "__main__":
    main()
