"""
Simulate multi-GPU DDP training on a single GPU.

Reads main_node_ddp.py, patches all CUDA device indices to 0 so every rank
runs on cuda:0, then exec's the modified source. All DDP logic (barriers,
rank guards, DistributedSampler, all_gather, etc.) still executes normally
because gloo supports multiple ranks on the same GPU (NCCL rejects duplicate GPUs).

Usage:
    torchrun --nproc_per_node=2 simulate_multigpu.py [same args as main_node_ddp.py]

Limitations:
    - All ranks share one GPU — ~Nx memory usage, reduce --batch_size to avoid OOM
    - Slower than single-process (context switching + gloo overhead on same device)
    - Not suitable for benchmarking, only for correctness testing of DDP code paths
"""

# Pre-import heavy libraries to avoid registration races between processes.
# Each torchrun worker is a separate process, so this runs once per process
# before the exec'd source tries to import them again (they'll be cached).
import torch  # noqa: F401
import torchvision  # noqa: F401
import torch_geometric  # noqa: F401

from pathlib import Path

source = Path(__file__).with_name("main_node_ddp.py").read_text()

# Patch 1: Device creation — remap cuda device to GPU 0
source = source.replace(
    'torch.device("cuda", local_rank)',
    'torch.device("cuda", 0)',
)

# Patch 2: GloveTextEmbedding device strings (appears in both rank 0 and non-rank-0 blocks)
source = source.replace(
    'f"cuda:{local_rank}"',
    '"cuda:0"',
)

# Patch 3: DDP wrapper device_ids
source = source.replace(
    "device_ids=[local_rank]",
    "device_ids=[0]",
)

# Patch 4: pynvml GPU handle
source = source.replace(
    "init_gpu_utilization(local_rank)",
    "init_gpu_utilization(0)",
)

# Patch 5: Use gloo backend — NCCL rejects multiple ranks on the same GPU
#          Also remove device_id= which is NCCL-only
source = source.replace(
    'backend="nccl", timeout=timedelta(hours=24), device_id=device',
    'backend="gloo", timeout=timedelta(hours=24)',
)

exec(compile(source, "main_node_ddp.py", "exec"))
