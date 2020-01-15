"""
    本部分主要包括pytorch的基础概念和语法
"""


import torch
import numpy as np

# 使用torch.empty可以返回填充了未初始化数据的张量，张量的形状由可变参数大小定义。
torch.empty(5, 3)

# 创建一个随机初始化矩阵
torch.rand(5, 3)

# 创建一个0填充的矩阵，制定数据类型为long
torch.zeros(5, 3, dtype = torch.long)

# 创建Tensor并使用现有数据初始化
x = torch.tensor([5.5, 3])

# 根据现有张量创建新张量，这个方法重用输入张量的属性，除非设置新的值进行覆盖
# new_*方法来创建对象
x = x.new_ones(5, 3, dtype = torch.double)

# 覆盖dtype，对象的size是相同的，只是值和类型发生了改变
x = torch.randn_like(x, dtype = torch.float)

# 获取张量的大小
# torch.size() 返回值是tuple类型，所以他支持tuple类型的所有操作。
print(x.size())

# 针对tensor的加法运算
y = torch.rand(5, 3)
x = torch.rand(5, 3)
x += y  # 第一种加法运算
print(x)

torch.add(x, y)  # 第二种加法运算

# 提供输出Tensor作为参数
result = torch.empty(5, 3)
torch.add(x, y, out = result)

# 替换
y.add_(x)  # 任何以下划线结尾的操作都会用结果替换原变量。例如：x.copy_(y), x.t_()


# 你可以使用Numpy索引方式相同的操作来进行对张量的操作
print('I want output')
print(x[:, 1])
print(x[1, :])

# torch.view可以改变张量的维度和大小
x = torch.randn(4, 4)
y = x.view(16)
z = x.view(-1, 8)

print(x.size(), y.size(), z.size())

# 如果张量只有一个元素，使用.item()来得到python数据类型的数值
x = torch.randn(1)

print(x, x.item())


# Numpy和Tensor的转换
a = torch.ones(5)
b = a.numpy()
print(a, b)

a.add_(1)
print(a, b)

# 使用from_numpy完成Numpy数组转换成Pytorch张量。
a = np.ones(5)
b = torch.from_numpy(a)
np.add(a, 1, out = a)
print('.............................')
print(a, b)


# CUDA张量：能够在GPU设备中运算的张量，使用.to方法可以将Tensor移动到GPU设备上。
# is_available 函数判断是否有GPU可以使用
if torch.cuda.is_available():
    device = torch.device("cuda")  # torch.device 将张量移动到指定的设备中
    y = torch.ones_like(x, device = device)  # 直接从GPU中创建张量
    x = x.to(device)
    z = x + y
    print(z)
    print(z.to("cpu", torch.double))  #  .to也会对变量的类型做更改