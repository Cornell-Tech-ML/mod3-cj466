import minitorch
import time
import numpy as np
#import matplotlib.pyplot as plt

FastTensorBackend = minitorch.TensorBackend(minitorch.FastOps)
GPUBackend = minitorch.TensorBackend(minitorch.CudaOps)


def run_matmul(backend: minitorch.TensorBackend, size: int = 16) -> None:
    """Run a matmul"""
    batch_size = 2

    x = minitorch.rand((batch_size, size, size), backend=backend)
    y = minitorch.rand((batch_size, size, size), backend=backend)
    _ = x @ y


if __name__ == "__main__":
    # Warmup
    run_matmul(FastTensorBackend)
    run_matmul(GPUBackend)

    ntrials = 3
    times = {}
    for size in [64, 128, 256, 512, 1024]:
        print(f"Running size {size}")
        times[size] = {}
        fast_times = []
        gpu_times = []
        for _ in range(ntrials):
            start_fast = time.time()
            run_matmul(FastTensorBackend, size)
            end_fast = time.time()

            start_gpu = time.time()
            run_matmul(GPUBackend, size)
            end_gpu = time.time()

            fast_time = end_fast - start_fast
            gpu_time = end_gpu - start_gpu

            fast_times.append(fast_time)
            gpu_times.append(gpu_time)

        times[size]["fast"] = np.mean(fast_times)
        times[size]["gpu"] = np.mean(gpu_times)
        print(times[size])

    print()
    print("Timing summary")
    for size, stimes in times.items():
        print(f"Size: {size}")
        for b, t in stimes.items():
            print(f"    {b}: {t:.5f}")

    sizes = list(times.keys())
    fast_times = [times[size]["fast"] for size in sizes]
    gpu_times = [times[size]["gpu"] for size in sizes]

"""     plt.figure(figsize=(10, 6))
    plt.plot(sizes, fast_times, marker='o', label='Fast Backend')
    plt.plot(sizes, gpu_times, marker='s', label='GPU Backend')

    plt.xlabel('Matrix Size', fontsize=12)
    plt.ylabel('Time (seconds)', fontsize=12)
    plt.title('Performance Comparison of Backends', fontsize=14)
    plt.xscale('log')
    plt.yscale('log')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.legend(fontsize=12)

    plt.savefig("comparison.png", dpi=300)
    plt.show() """