'''
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

'''


import torch
import torchvision
import matplotlib.pyplot as plt
import torch.utils.data
import new_cnn


Epoch=1         # 训练整批数据多少次, 为了节约时间, 我们只训练一次
Batch_Size=50
LR=0.01
Download_MNIST=False

train_transform=torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),   # 将numpy数据类型转化为Tensor
    # torchvision.transforms.Normalize([0.5],[0.5])
])

# 1.下载mnist数据集
# torch.Size([60000, 28, 28])
train_data=torchvision.datasets.MNIST(
    root='../../mofan/02/mnist',train=True,transform=train_transform,download=Download_MNIST
)

# torch.Size([10000, 28, 28])
test_data=torchvision.datasets.MNIST(
    root='../../mofan/02/mnist',train=False,transform=train_transform
)


'''显示数据图片'''
# print(train_data.train_data.size())     # (60000, 28, 28)
# print(train_data.train_labels.size())   # (60000)
# plt.imshow(train_data.train_data[0].numpy(), cmap='gray')   #转化成numpy数据显示
# plt.title('%i' % train_data.train_labels[0])    #%i和%d 没有区别。%i 是老式写法。都是整型格式
# plt.show()


# 2.批处理   (btach,1,28,28)
train_loader=torch.utils.data.DataLoader(dataset=train_data,batch_size=Batch_Size,shuffle=True)

test_loader=torch.utils.data.DataLoader(dataset=test_data,batch_size=Batch_Size,shuffle=False)


# 3.定义损失函数loss function 和优化方式（采用Adam）
model=new_cnn.MyCnn()
optimizer=torch.optim.Adam(model.parameters(),lr=LR)
loss_fuc=torch.nn.CrossEntropyLoss()

# 4.训练
for epoch in range(Epoch):
    sum_loss=0.0
    for step,(img,label) in enumerate(train_loader):
        # forward
        # img       torch.Size([50, 1, 28, 28])
        output=model(img)
        loss=loss_fuc(output,label)

        # 梯度清零
        optimizer.zero_grad()

        # backward
        loss.backward()
        optimizer.step()

        # 每训练100个batch打印一次平均loss
        sum_loss += loss.item()
        if step % 100 == 99:
            print('[%d, %d] loss: %.03f'
                  % (epoch + 1, step + 1, sum_loss / 100))
            sum_loss = 0.0

    # 每跑完一次epoch测试一下准确率
    total = 0
    correct_num = 0  # 正确的个数
    predict_list=[]

    for img, label in test_loader:

        # 前向传播得到结果
        out = model(img)

        # 取得分最高的那个类
        _, predicted = torch.max(out.data, 1)
        total += label.size(0)
        correct_num += (predicted == label).sum()

    print('第{}个epoch的识别准确率为：{:.6f}'.format(epoch + 1, (100 * correct_num / total)))
    torch.save(model.state_dict(), './model_{}.pth'.format(epoch + 1))

# 4.测试

test_d=torch.unsqueeze(test_data.test_data, dim=1).type(torch.FloatTensor)[:10]
print(test_d.size())
test_l=test_data.test_labels[:10]

test_output = model(test_d)
pred_y = torch.max(test_output, 1)[1].data.numpy().squeeze()
print(pred_y, 'prediction number')
print(test_l, 'real number')
"""
[7 2 1 0 4 1 4 9 5 9] prediction number
[7 2 1 0 4 1 4 9 5 9] real number
"""





