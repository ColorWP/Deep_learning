'''
logistic 逻辑回归
logistic回归简单的是做二分类问题，使用sigmoid函数将所有的正数和负数都变成0-1之间的数， 使用softmax激活函数 得到概率

train_transform=torchvision.transforms.Compose([
    torchvision.transforms.Scale(40), # 对图片的尺度进行缩小和放大 转化为40*40
    torchvision.transforms.RandomHorizontalFlip(p=0.5), #对图片进行概率为0.5的随机水平翻转
    torchvision.transforms.RandomCrop(32), # 对图片进行给定大小的随机裁剪 32*32
    torchvision.transforms.ColorJitter(brightness=1, contrast=2, saturation=3, hue=0),  # 给图像增加一些随机的光照
    torchvision.transforms.ToTensor(),   # 将numpy数据类型转化为Tensor
    torchvision.transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
])

test_transform=torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
])


torchvision.transforms.Scale(40)  新版本中已经更新 使用  torchvision.transforms.Resize(40)

最后一次得到的结果为   Test Loss: 0.268283, Acc: 0.926700

'''

import torch
import torchvision
import torch.utils.data

# 定义超参数
batch_size = 32
lr = 0.001
num_epoches = 20

# 对导入的图片进行处理
train_transform=torchvision.transforms.Compose([
    torchvision.transforms.ToTensor()
])

# gpu加速
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# mnist路径
path='../../../../mofan/02/mnist'

# 下载训练集 MNIST 手写数字训练集
train_dataset = torchvision.datasets.MNIST(root=path, train=True,
                               transform=train_transform,
                               download=False)
test_dataset = torchvision.datasets.MNIST(root=path, train=False,
                              transform=train_transform,
                              download=False)


# 加载数据集
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)



# 框架 Logstic_Regression
class MyNet(torch.nn.Module):
    def __init__(self, in_dim, n_class):
        super(MyNet, self).__init__()
        self.logstic = torch.nn.Linear(in_dim, n_class)

    def forward(self, x):
        out = self.logstic(x)
        return out


model = MyNet(28*28, 10).to(device)  # 图片大小是28x28

# 损失及优化
criterion = torch.nn.CrossEntropyLoss()  # 多分类用的交叉熵损失函数
optimizer = torch.optim.Adam(model.parameters(), lr=lr)


# 训练
for epoch in range(num_epoches):
    print('epoch {}'.format(epoch+1))
    print('-'*20)
    running_loss = 0.0
    running_acc = 0.0
    for i, data in enumerate(train_loader, 1):  # 表示i=1
        img, label = data  # torch.Size([32, 1, 28, 28])   torch.Size([32])

        img = img.view(img.size(0), -1)  # 将图片展开成 28x28 torch.Size([32, 784])

        # gpu
        img = img.to(device)
        label = label.to(device)

        # 向前传播
        out = model(img)
        loss = criterion(out, label)

        running_loss += loss.data.item() * label.size(0)
        _, pred = torch.max(out, 1)

        num_correct = (pred == label).sum()
        running_acc += num_correct.data.item()

        # 向后传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 每300次输出一次
        if i % 300 == 0:
            print('[{}/{}] Loss: {:.6f}, Acc: {:.6f}'.format(
                epoch + 1, num_epoches, running_loss / (batch_size * i),
                running_acc / (batch_size * i)))

    # 每一个epoch 输出一次
    print('Finish {} epoch, Loss: {:.6f}, Acc: {:.6f}'.format(
        epoch + 1, running_loss / (len(train_dataset)), running_acc / (len(
            train_dataset))))


    # 测试
    model.eval()
    eval_loss = 0.
    eval_acc = 0.
    for data in test_loader:
        img, label = data
        img = img.view(img.size(0), -1)

        '''
        （1）requires_grad=Fasle时不需要更新梯度， 适用于冻结某些层的梯度；
        
        （2）volatile=True相当于requires_grad=False，适用于推断阶段，不需要反向传播。
            这个现在已经取消了，使用with torch.no_grad()来替代，
        '''

        img = torch.autograd.Variable(img).to(device)
        label = torch.autograd.Variable(label).to(device)

        out = model(img)
        loss = criterion(out, label)
        eval_loss += loss.data.item() * label.size(0)
        _, pred = torch.max(out, 1)

        num_correct = (pred == label).sum()
        eval_acc += num_correct.data.item()
    print('Test Loss: {:.6f}, Acc: {:.6f}'.format(eval_loss / (len(
        test_dataset)), eval_acc / (len(test_dataset))))
    # print('Time:{:.1f} s'.format(time.time() - since))
    print()

# 保存模型
torch.save(model.state_dict(), './mynet.pth')




