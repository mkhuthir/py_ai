#!/usr/bin/python3

import torch

print("Checking your CUDA...")

if torch.cuda.is_available():
    print("CUDA is availble")
else:
    print("Sorry no CUDA")
