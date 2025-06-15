import timeit
import gc

import numpy as np
import torch

from cs336_basics.model import BasicsTransformerLM
from cs336_basics.nn_utils import cross_entropy


def benchmark(
    hyperparameters: dict[str, int],
    warm_up_steps: int,
    steps: int,
    device: torch.device,
    batch_size: int = 4,
    include_backward: bool = False,
    mixed_precision: bool = False,
) -> dict[str, dict[str, float]]:
    """
    Benchmark the performance of the transformer model.
    """
    device_type = device.type if device.type in ["cuda", "mps"] else "cpu"

    if mixed_precision:
        if device.type == "cuda" and torch.cuda.is_bf16_supported():
            autocast_dtype = torch.bfloat16
            print("Using BF16 autocast")
        elif device.type == "cuda":
            autocast_dtype = torch.float16
            print("Using FP16 autocast (BF16 not supported)")
        elif device.type == "cpu":
            autocast_dtype = torch.bfloat16
            print("Using BF16 autocast on CPU")
        else:
            autocast_dtype = torch.float16
            print("Using FP16 autocast")

    model = BasicsTransformerLM(**hyperparameters).to(device)

    data = torch.randint(
        0, hyperparameters["vocab_size"], size=(batch_size, hyperparameters["context_length"]), device=device
    )

    if include_backward:
        labels = torch.randint(
            0, hyperparameters["vocab_size"], size=(batch_size, hyperparameters["context_length"]), device=device
        )

    for _ in range(warm_up_steps):
        if include_backward:
            logits = model(data)
            loss = cross_entropy(logits, labels)
            loss.backward()
        else:
            with torch.no_grad():
                model(data)
        if data.is_cuda:
            torch.cuda.synchronize()

    forward_times = []
    backward_times = []
    total_times = []
    for _ in range(steps):
        step_start = timeit.default_timer()

        if include_backward:
            forward_start = timeit.default_timer()
            if mixed_precision:
                with torch.autocast(device_type=device_type, dtype=autocast_dtype):
                    logits = model(data)
            else:
                logits = model(data)
            if data.is_cuda:
                torch.cuda.synchronize()
            forward_end = timeit.default_timer()
            forward_times.append(forward_end - forward_start)

            loss = cross_entropy(logits, labels)

            backward_start = timeit.default_timer()
            loss.backward()
            if data.is_cuda:
                torch.cuda.synchronize()
            backward_end = timeit.default_timer()
            backward_times.append(backward_end - backward_start)
        else:
            with torch.no_grad():
                forward_start = timeit.default_timer()
                if mixed_precision:
                    with torch.autocast(device_type=device_type, dtype=autocast_dtype):
                        model(data)
                else:
                    model(data)
                if data.is_cuda:
                    torch.cuda.synchronize()
                forward_end = timeit.default_timer()
                forward_times.append(forward_end - forward_start)

        step_end = timeit.default_timer()
        total_times.append(step_end - step_start)

    results = {
        "forward_time": {"mean": float(np.mean(forward_times)), "std": float(np.std(forward_times))},
        "total_time": {"mean": float(np.mean(total_times)), "std": float(np.std(total_times))},
    }
    if include_backward:
        results["backward_time"] = {"mean": float(np.mean(backward_times)), "std": float(np.std(backward_times))}

    return results


if __name__ == "__main__":
    hyperparameters = {
        "vocab_size": 16384,
        "context_length": 512,
        "d_model": 1024,
        "num_layers": 32,
        "num_heads": 16,
        "d_ff": 4096,
        "rope_theta": 10000,
    }

    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using CUDA device")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS device")
    else:
        device = torch.device("cpu")
        print("Using CPU device")
    device = torch.device("cpu")

    batch_size = 2
    results = benchmark(
        hyperparameters,
        warm_up_steps=5,
        steps=10,
        device=device,
        batch_size=batch_size,
        include_backward=True,
        mixed_precision=True,
    )

    print("Benchmark Results:")
    print("-" * 40)
    for metric, stats in results.items():
        print(f"{metric}: {stats['mean']:.6f} ± {stats['std']:.6f} seconds")

    total_mean = results["total_time"]["mean"]

    # Tokens per second calculation
    tokens_per_step = batch_size * hyperparameters["context_length"]  # batch_size * context_length
    tokens_per_second = tokens_per_step / total_mean
    print(f"\nTokens per second: {tokens_per_second:.0f}")
