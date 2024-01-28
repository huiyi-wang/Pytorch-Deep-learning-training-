import torch
x = torch.arange(4.0)
print(x)
x.requires_grad_(True)
y = 2 * torch.dot(x,x)
print(y)
y.backward()

print(x.grad)
##梯度清零
x.grad.zero_()
print(x.grad)
y = x.sum()
y.backward()
print(x.grad)





