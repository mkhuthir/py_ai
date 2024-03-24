#!/usr/bin/python3

import torch
import numpy as np

# Tensor to Numpy
a = torch.ones(5)
b = a.numpy()
print(a)
print(b)
