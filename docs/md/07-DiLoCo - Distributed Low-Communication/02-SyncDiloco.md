# Synchronous DiLoCo

At a high level, the pseudo-code for implementing synchronous DiLoCo is as follows:

```python
# 1) Build your local model as usual:
local_model = GPT(config)
local_optimizer = AdamW(local_model, config)

# 2) Create separate "outer" parameters on CPU for aggregation:
outer_params = {}
for (name, local_param) in local_model.named_parameters():
    outer_params[name] = new_parameter_on_cpu( local_param.shape )

# 3) Create an "outer" optimizer, e.g. SGD:
outer_optimizer = SGD(outer_params, config)

# 4) Create PCCL communicator and connect:
communicator = Communicator(host=config["ccoip_host"], port=48148)
communicator.connect()

# 5) Build a SharedState containing all relevant parameters (model + any momentum buffers, etc.):
shared_state_dict = collect_all_params_and_buffers(local_model, outer_optimizer)
tensor_info_entries = [TensorInfo.from_torch(tensor, name) for (name, tensor) in shared_state_dict.items()]
shared_state = SharedState(tensor_info_entries)
shared_state.revision = 0  # e.g., the global "iteration number"

# TRAINING LOOP (SIMPLIFIED)
for outer_iter in range(MAX_OUTER_ITERS):

    # (A) Join aggregator "topology" via PCCL
    communicator.update_topology()

    sync_info = communicator.sync_shared_state(shared_state)

    # Copy aggregator data from shared_state into your local model if needed:
    #   e.g., for each "outer_param" in shared_state_dict, copy to local_model
    read_outer_params_into_local(local_model, outer_params)

    # (C) Perform local (inner) steps on random mini-batches
    for _ in range(config["inner_steps"]):
        loss = local_train_step(local_model, local_optimizer, batch_data())

    # (D) Compute difference: (outer_param - local_param)
    #     Then reduce among all workers => aggregator step
    outer_grad_list = []
    for name in outer_params:
        # CPU param minus local GPU param => gradient
        outer_params[name].grad = (outer_params[name] - local_model[name].cpu_data())

        outer_grad_list.append(outer_params[name].grad)

    # Use PCCL all_reduce_multiple_with_retry to average these differences:
    success, tx_bytes, rx_bytes = all_reduce_multiple_with_retry(
        communicator,
        outer_grad_list,
        ReduceOp.AVG
    )

    # Then outer optimizer steps:
    outer_optimizer.step()
    outer_optimizer.zero_grad()

    # Finally copy aggregated outer params back into local model:
    copy_outer_params_to_local(local_model, outer_params)

    # (E) Increment shared_state.revision and loop
    shared_state.revision += 1
    if termination_condition(...):
        break
```

A full example implementing synchronous DiLoCo is available here: https://github.com/PrimeIntellect-ai/pccl/blob/main/python/examples/nanogpt_diloco/sync_diloco.py

A full example implementing synchronous DiLoCo combined with FSDP (Fully Sharded Data Parallel) is available here: https://github.com/PrimeIntellect-ai/pccl/blob/main/python/examples/nanogpt_diloco/sync_diloco_fsdp.py
