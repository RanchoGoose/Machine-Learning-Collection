import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
my_tensor = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float32,
                          device=device, requires_grad=True)



x = torch.empty(size=(1, 5)).normal_(mean=0, std=1)

import numpy as np


batch = 32
n = 10
m = 20
p = 30

tensor1 = torch.rand((batch, n, m))
tensor2 = torch.rand((batch, m, p))
our_bmm = torch.bmm(tensor1, tensor2)

x1 = torch.rand((5, 5))
x2 = torch.rand((1, 5))

z = x1 - x2

z = torch.clamp(x, min=0, max=10)



x = torch.rand((3, 5))
rows = torch.tensor([1, 0])
cols = torch.tensor([4, 0])
print(x[rows, cols].shape)

print(torch.where(x > 5, x, x*2))