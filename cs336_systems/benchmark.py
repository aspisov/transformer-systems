import timeit

import numpy as np
import torch
from cs336_basics.model import BasicsTransformerLM
from cs336_basics.nn_utils import cross_entropy


def benchmark(
    hyperparameters: dict[str, int],
    warm_up_steps: int,
    steps: int,
    batch_size: int = 4,
    device: torch.device | None = None,
    include_backward: bool = False,
) -> dict[str, dict[str, float]]:
    """
    Benchmark the performance of the transformer model.

    Args:
        hyperparameters: Model hyperparameters dictionary
        warm_up_steps: Number of warm-up steps before timing
        steps: Number of steps to time
        batch_size: Batch size for random data
        device: Device to run benchmark on
        include_backward: Whether to include backward pass in timing

    Returns:
        dict: Dictionary containing benchmark metrics where each value is a dict with 'mean' and 'std':
            - forward_time: {"mean": mean_time, "std": std_time} for forward pass
            - total_time: {"mean": mean_time, "std": std_time} for total step time
            - backward_time: {"mean": mean_time, "std": std_time} for backward pass (if include_backward=True)
    """
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
        "vocab_size": 10000,
        "context_length": 256,
        "d_model": 512,
        "num_layers": 4,
        "num_heads": 16,
        "d_ff": 1344,
        "rope_theta": 10000,
    }

    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    results = benchmark(hyperparameters, 1, 5, device=device, batch_size=4, include_backward=True)

    print("Benchmark Results:")
    print("-" * 40)
    for metric, stats in results.items():
        print(f"{metric}: {stats['mean']:.6f} ± {stats['std']:.6f} seconds")

    total_mean = results["total_time"]["mean"]

    # Tokens per second calculation
    tokens_per_step = 4 * hyperparameters["context_length"]  # batch_size * context_length
    tokens_per_second = tokens_per_step / total_mean
    print(f"\nTokens per second: {tokens_per_second:.0f}")
