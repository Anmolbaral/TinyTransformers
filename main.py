from torch import torch

x = torch.tensor([[1, 2], [4, 5]])

y = torch.matmul(x, x)
print(y)
