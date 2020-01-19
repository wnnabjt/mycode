import torch
from torch import nn, optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


batch_size = 128
learning_rate = 1e-2
num_epoches = 20

data_tf = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize([0.5], [0.5])])

train_dataset = datasets.MNIST(root='./data', train=True, transform=data_tf, download=True)
test_dataset = datasets.MNIST(root='./data', train=False, transform=data_tf)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


class Batch_Net(nn.Module):

    def __init__(self, in_dim, n_hidden_1, n_hidden_2, out_dim):
        super(Batch_Net, self).__init__()
        self.layer1 = nn.Linear(in_dim, n_hidden_1)
        self.layer2 = nn.Linear(n_hidden_1, n_hidden_2)
        self.layer3 = nn.Linear(n_hidden_2, out_dim)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x



model = Batch_Net(28 * 28, 300, 100, 10)
if torch.cuda.is_available():
    model = model.cuda()

# 定义损失函数和优化方法
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr = learning_rate)

# 开始训练
for epoch in range(num_epoches):
    for data in train_loader:
        img, label = data
        print(img.size(0))
        img = img.view(img.size(0), -1)
        img, label = Variable(img).cuda(), Variable(label).cuda()
        # =================forward====================#
        out = model(img)
        loss = criterion(out, label)
        print_loss = loss.data.item()
        # ================backward====================#
        optimizer.zero_grad() # 反向传播之前需要梯度清零，不然会造成梯度的累加
        loss.backward()
        optimizer.step()
        # if epoch % 50 == 0:
        #     print('epoch: {}, loss: {:.4}'.format(epoch + 1, loss.data.item()))
    # 测试训练集
    model.eval()
    eval_loss = 0
    eval_acc = 0
    for data in test_loader:
        img, label = data
        img = img.view(img.size(0), -1)
        img = Variable(img, volatile = True).cuda()
        label = Variable(label, volatile = True).cuda()
        out = model(img)
        loss = criterion(out, label)
        eval_loss += loss.item() * label.size(0)
        _, pred = torch.max(out, 1)
        num_correct = (pred == label).sum()
        eval_acc += num_correct.item()
    print('Test Loss: {:.6f}'.format(
        eval_loss / (len(test_dataset)),
        eval_acc / (len(test_dataset))
    ))