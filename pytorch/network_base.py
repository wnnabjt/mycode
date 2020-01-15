"""

"""


import torch

x = torch.ones(2, 2, requires_grad = True)
print(x)

y = x + 2
print(y)

print(y.grad_fn)
print('.++++++++++++++.')
z = y * y * 3
out = z.mean()
print(z, out)

a = torch.randn(2, 2)
a = ((a * 3) / (a - 1))
print(a.requires_grad)
a.requires_grad_(True)  # 可以通过.requires_grad_()来改变requires_grad的属性
print(a.requires_grad)
b = (a * a).sum()
print(b.grad_fn)


print(x.requires_grad)
print((x ** 2).requires_grad)

with torch.no_grad():  # 如果.requires_grad = True 但是你又不希望进行Autograd的计算，那么可以将变量包裹在with torch.no_grad()中
    print((x ** 2).requires_grad)
