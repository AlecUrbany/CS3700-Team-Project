import torch
import numpy as np

# Tensor Initialization
data = [[1,2],[3,4]]
x_data = torch.tensor(data)

np_array = np.array(data)
x_np = torch.from_numpy(np_array)

x_ones = torch.ones_like(x_data)
print(f"Ones Tensor: \n {x_ones} \n")

x_rand = torch.rand_like(x_data, dtype=torch.float)
print(f"Random Tensor: \n {x_rand} \n")

# Tensor Attributes
tensor = torch.rand(3, 4)
print(tensor)
print(f"Shape of tensor: {tensor.shape}")
print(f"Datatype of tensor: {tensor.dtype}")
print(f"Device tensor is stored on: {tensor.device} \n")

# Tensor Operations
# All tensor operations can be run in the GPU
if torch.cuda.is_available():
    tensor = tensor.to('cuda')
    print(f"Device tensor is stored on: {tensor.device} \n")

# Specify index
tensor = torch.ones(4,4)
tensor[:,1] = 0
print(tensor, "\n")

# Concatenating tensors
t1 = torch.cat([tensor, tensor, tensor], dim=1)
print(f"{t1} \n")

data = [[3, 1, 3, 4],
        [2, 1, 2, 2],
        [0, 1, 1, 2],
        [2, 2, 4, 3]]
        
# Multiplication
tensor = torch.tensor(data)
print(f"tensor.mul(tensor) \n {tensor.mul(tensor)} \n")
print(f"tensor * tensor \n {tensor * tensor} \n")

# Matrix Multiplication
print(f"tensor.matmul(tensor.T) \n {tensor.matmul(tensor.T)} \n")
print(f"tensor @ tensor.T \n {tensor @ tensor.T} \n")

# In-place operations use the "_"
tensor = torch.ones(4,4)
tensor[:,1] = 0
print({tensor}, "\n")
tensor.add_(5)
print(tensor, "\n")









