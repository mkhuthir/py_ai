#!/usr/bin/python3

import torch

print("\n1D Tensors")

# Create a Tensor with just ones in a column
a = torch.ones(5)
print(a)

# Create a Tensor with just zeros in a column
b = torch.zeros(5)
print(b)

# Create a Tensor with custom values
c = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
print(c)

# Create a 2D Tensor with just zeros 
print("\n2D Tensors")
d = torch.zeros(3,2)
print(d)

# Create a 2D Tensor with just ones
e = torch.ones(3,2)
print(e)

# Create a 2D Tensor with random values 
f = torch.rand(5,5)
print(f)

# Create a 3D Tensor with custom values
print("\n3D Tensors")
g = torch.tensor([
                    [   [1, 2],
                        [3, 4]
                    ],

                    [   [5, 6], 
                        [7, 8]
                    ]
                ])
print(g)
 
