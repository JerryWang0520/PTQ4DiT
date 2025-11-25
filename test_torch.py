import torch

x = torch.tensor(1.23)
print(x)
print(list(x.shape))


# Test on GPU tensor
x = torch.tensor([1.5, 2.3, 3.7]).cuda()

# This transfers each element to CPU
for val in x:
    python_val = val.item()  # GPU→CPU transfer happens here
    print(type(python_val))  # <class 'float'> (Python native type)

# Alternative: keep on GPU
for val in x:
    gpu_val = val  # Still a GPU tensor
    print(type(gpu_val))  # <class 'torch.Tensor'> (GPU tensor)
