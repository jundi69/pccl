# Rank 1
PCCL_LOG_LEVEL=DEBUG;BASE_PORT=10000;MASTER_PORT=10000 python sync_diloco_fsdp.py --nproc-per-node=2 --master_port=10000 /home/mike/CLionProjects/pccl/python/examples/nanogpt_diloco/sync_diloco_fsdp.py --config default_config_other.json &

# Rank 2
PCCL_LOG_LEVEL=DEBUG;BASE_PORT=20000;MASTER_PORT=20000 python sync_diloco_fsdp.py --nproc-per-node=2 --master_port=20000 /home/mike/CLionProjects/pccl/python/examples/nanogpt_diloco/sync_diloco_fsdp.py --config default_config_other.json &


