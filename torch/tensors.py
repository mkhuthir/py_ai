#!/usr/bin/python3

import torch

# Create a Tensor with just ones in a column
a = torch.ones(5)
print("\n1D Tensors")
print(a)

# Create a Tensor with just zeros in a column
b = torch.zeros(5)
print("\nZeros in a column")
print(b)

# Create a Tensor from data
c = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
print("\nTensor from data")
print(c)

# Create a 2D Tensor with just zeros 
print("\n2D Tensor with just zeros")
d = torch.zeros(3,2)
print(d)

# Create a 2D Tensor with just ones
e = torch.ones(3,2)
print("\n2D Tensor with just ones")
print(e)

# Create a 2D Tensor with random values 
f = torch.rand(5,5)
print("\n2D Tensor with random values ")
print(f)

# Create a 3D Tensor with custom values
g = torch.tensor([
                    [   [1, 2], [3, 4]],
                    [   [5, 6], [7, 8]]
                ])
print("\n3D Tensor with custom values")
print(g)
 
# Construct a 3x3x3 matrix, uninitialized:
print("\n3x3x3 matrix, uninitialized")
h = torch.empty(3, 3, 3)
print(h)

