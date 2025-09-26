import torch_directml
import torch

dml = torch_directml.device()
x = torch.rand(3, 3, device=dml)
y = torch.rand(3, 3, device=dml)
print(x @ y)