import os
from dataclasses import dataclass

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from cs336_basics.data import get_batch
from cs336_basics.model import BasicsTransformerLM
from cs336_basics.nn_utils import cross_entropy
from cs336_basics.optimizer import AdamW

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
    context_length: int
    vocab_size: int
    num_iterations: int
    batch_size: int
    device: str


def setup(rank, world_size, device):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    if device == "cuda":
        dist.init_process_group("nccl", rank=rank, world_size=world_size)
    else:
        dist.init_process_group("gloo", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


def train(rank: int, world_size: int, config: Config):
    setup(rank, world_size, config.device)

    if config.device == "cuda":
        torch.cuda.set_device(rank)
        device = f"cuda:{rank}"
    else:
        device = config.device

    dataset = np.random.randint(0, config.vocab_size, 2**27)
    model = BasicsTransformerLM(**hyperparameters, rope_theta=10000).to(device)
    optimizer = AdamW(model.parameters())

    for step in range(config.num_iterations):
        x, y = get_batch(
            dataset,
            batch_size=config.batch_size // world_size,
            context_length=config.context_length,
            device=device,
        )

        logits = model(x)

        loss = cross_entropy(logits, y)
        loss.backward()

        for param in model.parameters():
            dist.all_reduce(tensor=param.grad, op=dist.ReduceOp.AVG, async_op=False)

        if rank == 0:
            print(f"Step {step}, Loss: {loss.item():.4f}")

        optimizer.step()
        optimizer.zero_grad()

    cleanup()


def main(world_size: int = 4):
    config = Config(vocab_size=16384, context_length=128, num_iterations=100, batch_size=8, device="cpu")

    mp.spawn(fn=train, args=(world_size, config), nprocs=world_size, join=True)


if __name__ == "__main__":
    main()
