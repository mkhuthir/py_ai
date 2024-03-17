#!/usr/bin/python3

import torch

print("Checking your CUDA...")

if torch.cuda.is_available():
    print("CUDA is availble")

    # Create a tensor for CPU
    tensor_cpu = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], device='cpu')
 
    # Create a tensor for GPU
    tensor_gpu = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], device='cuda')

    # This uses CPU
    tensor_cpu = tensor_cpu * 5
    
    # This uses GPU
    tensor_gpu = tensor_gpu * 5

    print(tensor_cpu)
    print(tensor_gpu)

    # Move GPU tensor to CPU
    tensor_gpu_cpu = tensor_gpu.to(device='cpu')
    
    # Move CPU tensor to GPU
    tensor_cpu_gpu = tensor_cpu.to(device='cuda')

else:
    print("Sorry no CUDA")

    # Create a tensor for CPU
    tensor_cpu = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], device='cpu')

    # This uses CPU
    tensor_cpu = tensor_cpu * 5

    print(tensor_cpu)