# Async DiLoCo

Async DiLoCo is a more advanced version of DiLoCo which hides the communication time associated with the pseudo gradient reduction by
overlapping it with local training steps of the next step.
This technically applies slightly dirty parameter updates.
Given that each step is effectively an N=2 pipeline, we need to take special care of newcomers to properly insert the into the pipeline.
Specifically, given that the all reduce that completes at any given outer step `t` has commenced before the newcomer has joined, we need to ensure that the newcomer
gets to "eavesdrop" on the result, which only the pre-existing workers will see by performing a second shared state synchronization.


## Simplified pseudo code
```python
# 1) Build your local model as usual:
local_model = GPT(config)
local_optimizer = AdamW(local_model, config)

# 2) Create separate "outer" parameters on CPU for aggregation:
outer_params = {}
for (name, local_param) in local_model.named_parameters():
    outer_params[name] = new_parameter_on_cpu(local_param.shape)

# 3) Create an "outer" optimizer, e.g. SGD:
outer_optimizer = SGD(outer_params, config)

# 4) Create PCCL communicator and connect:
communicator = Communicator(host=config["ccoip_host"], port=0)
communicator.connect()

# 5) Build a SharedState for the aggregator (outer params + momentum buffers), and set revision=0
shared_state_dict = collect_outer_params_and_buffers(outer_params, outer_optimizer)
tensor_info_entries = [TensorInfo.from_torch(tensor, name) for (name, tensor) in shared_state_dict.items()]
shared_state = SharedState(tensor_info_entries)
shared_state.revision = 0

# 6) Prepare structures for asynchronous all-reduce:
all_reduce_thread = None
last_pseudo_grads = None

# 7) The asynchronous training loop
for outer_iter in range(MAX_OUTER_ITERS):

    # (A) Possibly update topology if new peers are pending
    #     - If we detect pending peers, we must first join any running all-reduce thread
    #       so that update_topology won't conflict with an in-flight collective
    if communicator.are_peers_pending():
        if all_reduce_thread is not None:
            all_reduce_thread.join()  # Wait for the last reduce to finish
        communicator.update_topology()
        topology_updated = True
    else:
        topology_updated = False

    # If topology was updated, sync aggregator state so new peers can catch up
    if topology_updated:
        communicator.sync_shared_state(shared_state)
        copy_outer_params_into_local(local_model, outer_params)

    # (B) If there's a previous all-reduce in progress, wait for it to finish.
    #     Then apply the averaged “pseudo-grads” to outer_params and copy aggregator back locally.
    if all_reduce_thread is not None:
        all_reduce_thread.join()

        for name in outer_params:
            outer_params[name].grad = last_pseudo_grads[name]
        outer_optimizer.step()
        outer_optimizer.zero_grad()

        copy_outer_params_into_local(local_model, outer_params)

        # If new peers arrived mid-run, re-sync aggregator so they obtain the updated state
        if topology_updated and outer_iter > 0:
            communicator.sync_shared_state(shared_state)

    # (C) Perform local “inner” steps:
    for _ in range(config["inner_steps"]):
        local_loss = local_train_step(local_model, local_optimizer, batch_data())

    # (D) Compute new pseudo-grads = (outer_param - local_param).
    #     If no aggregator step was done yet (meaning can_outer_step == False in the code),
    #     and topology updated, then we must do a 'late_joiner' sync + reset local model to aggregator.
    can_outer_step = (all_reduce_thread is not None)
    if not can_outer_step and topology_updated and outer_iter > 0:
        #  - We are the late joiner, so we do another sync to get aggregator
        #  - Then we reset local parameters to that aggregator state
        communicator.sync_shared_state(shared_state)
        copy_outer_params_into_local(local_model, outer_params)
    
    pseudo_grad_dict = {}
    for name, local_param in local_model.named_parameters():
        diff = outer_params[name] - local_param.cpu()
        pseudo_grad_dict[name] = diff

    # (E) Launch a background thread to all-reduce these pseudo-grads
    def async_all_reduce(grad_dict):
        all_reduce_multiple_with_retry(communicator, list(grad_dict.values()), op=ReduceOp.AVG)

    last_pseudo_grads = pseudo_grad_dict
    all_reduce_thread = threading.Thread(
        target=async_all_reduce,
        args=(last_pseudo_grads,)
    )
    all_reduce_thread.start()

    # (F) Bump shared_state revision; possibly log, check for termination, etc.
    shared_state.revision += 1
    if termination_condition():
        break

communicator.close()
```

A full example implementing async DiLoCo is available here: https://github.com/PrimeIntellect-ai/pccl/blob/main/python/examples/nanogpt_diloco/async_diloco.py
