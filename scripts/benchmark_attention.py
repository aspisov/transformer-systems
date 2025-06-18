import gc
import timeit

import torch
from cs336_basics.model import scaled_dot_product_attention


def print_memory_usage():
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3  # GB
        reserved = torch.cuda.memory_reserved() / 1024**3  # GB
        print(f"CUDA Memory - Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB")


def benchmark_attention(use_backward: bool = True):
    device = "cuda"
    dtype = torch.bfloat16
    batch_size = 4

    num_heads = 1
    d_models = [16, 32, 64, 128]
    seq_lengths = [256, 1024, 4096, 8192, 16384]

    for d_model in d_models:
        for seq_len in seq_lengths:
            print(f"d_model: {d_model}, seq_len: {seq_len}")
            forward_times = []
            backward_times = []

            Q = torch.randn(batch_size, seq_len, d_model, device=device, dtype=dtype, requires_grad=use_backward)
            K = torch.randn(batch_size, seq_len, d_model, device=device, dtype=dtype, requires_grad=use_backward)
            V = torch.randn(batch_size, seq_len, d_model, device=device, dtype=dtype, requires_grad=use_backward)

            for _ in range(10):
                out = scaled_dot_product_attention(Q, K, V)
                if use_backward:
                    loss = out.sum()
                    loss.backward()
                    Q.grad = None
                    K.grad = None
                    V.grad = None

            # Forward timing
            for _ in range(100):
                with torch.no_grad():
                    start = timeit.default_timer()
                    out = scaled_dot_product_attention(Q, K, V)
                    torch.cuda.synchronize()
                    end = timeit.default_timer()
                    forward_times.append(end - start)

            print(f"forward time: {sum(forward_times) / len(forward_times):.6f}s")
            print_memory_usage()

            # Backward timing
            if use_backward:
                for _ in range(100):
                    start = timeit.default_timer()
                    out = scaled_dot_product_attention(Q, K, V)
                    loss = out.sum()
                    loss.backward()
                    Q.grad = None
                    K.grad = None
                    V.grad = None
                    torch.cuda.synchronize()
                    end = timeit.default_timer()
                    backward_times.append(end - start)
                print(f"backward time: {sum(backward_times) / len(backward_times):.6f}s")
            print("-" * 120)

            del Q, K, V, out
            if "loss" in locals():
                del loss

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


if __name__ == "__main__":
    benchmark_attention(True)
