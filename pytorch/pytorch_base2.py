"""
    深度学习入门之PyTorch学习记录
"""

import torch
from torch import nn as nn
import numpy as np
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import torchvision.transforms as transforms


#  Tensor's create
a = torch.Tensor([[2, 3], [4, 8], [7, 9]])  # torch.Tensor 默认的是torch.FloatTensor数据类型
print('a is : {}'.format(a))  # 显示a中的元素
print('a size i s{}'.format(a.size()))  # 显示a的大小

b = torch.LongTensor([[2, 3], [4, 8], [7, 9]])  # 也可以创建其他类型的Tensor
print('b is : {}'.format(b))

c = torch.zeros((3, 2))  # 也可以创建一个全是零的空Tensor
print('zero tensor: {}'.format(c))

d = torch.randn((3, 2))  # 或者取一个正态分布做随机初始值


# Tensor's find
a[0, 1] = 100
print('changed a is : {}', format(a))

# Tensor's conver
numpy_b = b.numpy()  # Tensor to np
print('conver to numpy is \n {}'.format(numpy_b))

e = np.array([[2, 3], [4, 5]])
torch_e = torch.from_numpy(e)  # np to Tensor
print('from numpy to torch.Tensor is {}'.format(torch_e))
f_torch_e = torch_e.float()  # change data type
print('change data type to float tensor:{}'.format(f_torch_e))


#  Tensor in GPU
if torch.cuda.is_available():
    a = a.cuda()
    print(a)


# Variable : 提供了自动求导功能
"""
    autograd.Variable:
        data:tensor数值
        grad_fn：得到这个Variable的操作  # 比如是通过哪种运算得来的
        grad：这个Variable的反向传播梯度
"""


# Create Variable
x = Variable(torch.Tensor([1]), requires_grad = True)
w = Variable(torch.Tensor([2]), requires_grad = True)
b = Variable(torch.Tensor([3]), requires_grad = True)


# bulid a computational graph
y = w * x + b

# Compute gradients
# x.backward()
y.backward()  # same as y.backward(torch.FloatTensor([1]))
# print(x.grad_fn, w.grad_fn, b.grad_fn, y.grad_fn)
# print out the gradients
#  print(x, w, b, y)
print(x.grad)
print(w.grad)
print(b.grad)
# print(y.grad)

x = torch.randn(3)
x = Variable(x, requires_grad = True)

y = x * 2
print(y)

# 将其原本的梯度分别乘上1， 0,1和0.01
y.backward(torch.FloatTensor([1, 0.1, 0.01]))
print(x.grad) # output : 2.0000, 0.2000, 0.0200


#  Dataset

class MyDataset(Dataset):

    def __init(self, dataPath, transform = None, target_transform = None):
        imgsPath = open(dataPath, 'r')
        imgs = []
        for line in imgsPath:
            line = line.rstrip()  # 将字符串右边的默认字符去掉，默认为空格 str.rstrip(str = "").
            words = line.split()  # 将字符串按照某字符切片存储，默认为空格， tab， 换行，切片次数默认为全切 str.split(str = "", num = string.count(str)).
            imgs.append((words[0], int(words[1]))) # word[0] 为图片信息， word[1]为标签信息
            self.imgs = imgs
            self.transform = transform
            self.target_transform = target_transform

    def __getitem__(self, index):
        fn, label = self.imgs[index]
        img = Image.open(fn).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            label = self.transform(label)
        return img, label

    def __len__(self):
        return len(self.imgs)


# DataLoader

# DataLoader(dataset, batch_size = 1, shuffle = False, sampler = None, num_workers = 0, pin_memory = False, drop_last = False)
# shuffle:是否将数据打乱
# sampler：样本抽样
# num_workers:使用多进程加载的进程数，0表示不使用多进程
# pin_memory，是否将数据保存在pin memory区，pin memory中的数据转到GPU会快一些
# drop_last: dataset中的数据个数可能不是batch_size的整数倍，drop_last为True时会将多出来不足一个batch的数据丢弃


# ImageFolder
# dest = ImageFolder(root = 'root_path', transform = None, loader = default_loader)
# root是根目录，transform和target_transrform是图片增强， loader是将图片转换成我们需要的图片类型进入神经网络

# nn.Module(模组)
# 在pytorch里面编写神经网络，所有的层结构和损失函数都来自torch.nn，所有的模型构建都是从这个基类nn.Module继承的。

class Net_name(nn.Module):
    def __init__(self, other_arguments):
        super(Net_name, self).__init()
        # other network layer

    def forward(self, x):
        # ...
        return x

# 计算损失函数
criterion = nn.CrossEntropyLoss()
# loss = criterion(output, target)

# torch.optim
# optimizer = torch.optim.SGD(model.parameters(), lr = 0.01, momentum = 0.9)

