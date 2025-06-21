import os
from timeit import default_timer

import click
import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def setup(rank, world_size, device):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    if device == "cuda":
        dist.init_process_group("nccl", rank=rank, world_size=world_size)
    else:
        dist.init_process_group("gloo", rank=rank, world_size=world_size)


def distributed_benchmark(rank, world_size, warm_up_steps: int, steps: int, device: str, data_size: int):
    setup(rank, world_size, device)

    for step in range(warm_up_steps):
        data = torch.randn(data_size, device=device)
        dist.all_reduce(data, async_op=False)
        if device == "cuda":
            torch.cuda.synchronize()

    times = []

    for step in range(steps):
        data = torch.randn(data_size, device=device)
        start = default_timer()
        dist.all_reduce(data, async_op=False)
        if device == "cuda":
            torch.cuda.synchronize()
        end = default_timer()
        times.append(end - start)

    avg_time = sum(times) / len(times)
    bandwidth = (data_size * 4) / avg_time / 1e9

    if rank == 0:
        print(f"Size: {data_size * 4 / 1e9} GB, Time: {avg_time:.6f}s, Bandwidth: {bandwidth:.2f} GB/s")


@click.command()
@click.option("--world-size", type=int, default=4)
@click.option("--warm-up-steps", type=int, default=5)
@click.option("--steps", type=int, default=10)
@click.option("--device", type=str, default="cpu")
@click.option("--data-size", type=int, default=2**27)
def main(world_size: int, warm_up_steps: int, steps: int, device: str, data_size: int):
    if not torch.cuda.is_available():
        device = "cpu"
    print("Using", device)

    mp.spawn(
        fn=distributed_benchmark,
        args=(world_size, warm_up_steps, steps, device, data_size),
        nprocs=world_size,
        join=True,
    )


if __name__ == "__main__":
    main()
