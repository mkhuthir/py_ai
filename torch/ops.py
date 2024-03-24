#!/usr/bin/python3

import torch

# Create tensor
tensor1 = torch.tensor([[1,2,3],[4,5,6]])
tensor2 = torch.tensor([[-1,2,-3],[4,-5,6]])
print("\nt1 = ",tensor1)
print("\nt2 = ",tensor2)

# Addition
print("\nt1+t2 = ",tensor1+tensor2)

# We can also use
print("\nt1+t2 = ",torch.add(tensor1,tensor2))

# in place addition
tensor1.add_(tensor2)
print("\nt1 after add = ",tensor1)

# Subtraction
print("\nt1-t2 = ",tensor1-tensor2)
# We can also use
print("\nt1-t2 = ",torch.sub(tensor1,tensor2))
 
# Multiplication
# Tensor with Scalar
print("\nt1*2 = ",tensor1 * 2)
 
# Tensor with another tensor
# Element wise Multiplication
print("\nt1*t2 = ",tensor1 * tensor2)
 
# Matrix multiplication
tensor3 = torch.tensor([[1,2],[3,4],[5,6]])
print("\nt3 = ",tensor3)
print("\nt1 mm t3 = ",torch.mm(tensor1,tensor3))
 
# Division
# Tensor with scalar
print("\nt1/2 = ",tensor1/2)
 
# Tensor with another tensor
# Element wise division
print("\nt1/t2 = ",tensor1/tensor2)