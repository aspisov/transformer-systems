import os
from dataclasses import dataclass
from timeit import default_timer

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from cs336_basics.data import get_batch
from cs336_basics.model import BasicsTransformerLM
from cs336_basics.nn_utils import cross_entropy
from cs336_basics.optimizer import AdamW
from cs336_systems.ddp import DDP

hyperparameters = {
    "d_model": 768,
    "num_layers": 12,
    "num_heads": 12,
    "d_ff": 3072,
    "vocab_size": 16384,
    "context_length": 128,
}


@dataclass
class Config:
    world_size: int
    context_length: int
    vocab_size: int
    num_iterations: int
    batch_size: int
    device: str


def setup(rank, world_size, device):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29501"
    if device == "cuda":
        dist.init_process_group("nccl", rank=rank, world_size=world_size)
    else:
        dist.init_process_group("gloo", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


def train(rank: int, config: Config):
    setup(rank, config.world_size, config.device)

    if config.device == "cuda":
        torch.cuda.set_device(rank)
        device = f"cuda:{rank}"
    else:
        device = config.device

    dataset = np.random.randint(0, config.vocab_size, 2**27)
    model = BasicsTransformerLM(**hyperparameters, rope_theta=10000).to(device)
    ddp_model = DDP(model)
    optimizer = AdamW(ddp_model.parameters())

    for step in range(config.num_iterations):
        step_start = default_timer()
        x, y = get_batch(
            dataset,
            batch_size=config.batch_size // config.world_size,
            context_length=config.context_length,
            device=device,
        )

        logits = ddp_model(x)

        loss = cross_entropy(logits, y)
        loss.backward()

        comm_start = default_timer()

        ddp_model.finish_gradient_synchronization()

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        comm_end = default_timer()

        optimizer.step()
        optimizer.zero_grad()
        step_end = default_timer()
        if rank == 0:
            print(f"Step {step} took {step_end - step_start:.2f} seconds")
            print(f"Communication took {comm_end - comm_start:.2f} seconds")
            print(f"Communication overhead: {(comm_end - comm_start) / (step_end - step_start) * 100:.2f}%")

    cleanup()


def main():
    config = Config(world_size=2, vocab_size=16384, context_length=128, num_iterations=5, batch_size=32, device="cuda")

    mp.spawn(fn=train, args=(config,), nprocs=config.world_size, join=True)




if __name__ == "__main__":
    main()


