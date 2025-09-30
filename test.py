import torch

# This will print the device currently set for PyTorch operations
# 'cuda' means your ROCm-enabled device, 'cpu' means the CPU.
print(f"PyTorch is currently set to use device: {torch.cuda.current_device()}")
print(f"Is ROCm/CUDA available? {torch.cuda.is_available()}")