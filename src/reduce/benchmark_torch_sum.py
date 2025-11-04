import torch
import time
import numpy as np

def benchmark_torch_sum():
    # Same size as your CUDA code
    size = 2048 * 250_000
    num_runs = 50
    
    # CPU timing
    cpu_tensor = torch.ones(size, dtype=torch.float32)
    
    # Warmup
    _ = torch.sum(cpu_tensor)
    
    start_time = time.perf_counter()
    for _ in range(num_runs):
        result = torch.sum(cpu_tensor)
    end_time = time.perf_counter()
    
    cpu_time = (end_time - start_time) * 1000 / num_runs  # Convert to ms
    print(f"PyTorch CPU sum average time: {cpu_time:.4f} ms")
    print(f"PyTorch CPU result: {result.item()}")
    
    # GPU timing (if CUDA available)
    if torch.cuda.is_available():
        gpu_tensor = torch.ones(size, dtype=torch.float32, device='cuda')
        
        # Warmup
        _ = torch.sum(gpu_tensor)
        torch.cuda.synchronize()
        
        # Use CUDA events for accurate timing
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        start_event.record()
        for _ in range(num_runs):
            result = torch.sum(gpu_tensor)
        end_event.record()
        torch.cuda.synchronize()
        
        gpu_time = start_event.elapsed_time(end_event) / num_runs
        print(f"PyTorch GPU sum average time: {gpu_time:.4f} ms")
        print(f"PyTorch GPU result: {result.item()}")
    else:
        print("CUDA not available for PyTorch")

if __name__ == "__main__":
    benchmark_torch_sum()
