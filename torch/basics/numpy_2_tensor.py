#!/usr/bin/python3

import torch
import numpy as np

# Numpy to Tensor
a = np.ones(5)
b = torch.from_numpy(a)
print(a)
print(b)

# Add operation
np.add(a, 1, out=a)
print(a)
print(b)


