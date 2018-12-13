'''
生成对抗神经网络gan
mnist 28*28
全连接

40分钟左右

'''

import torch
import torch.utils.data
import torchvision
import os
import argparse
from datetime import datetime


# 1.定义超参数
parser=argparse.ArgumentParser()
parser.add_argument('--batch_size',type=int,default=200,help='Batch processing')
parser.add_argument('--epoch',type=int,default=100,help='All data trained several times')
parser.add_argument('--nz',type=int,default=100,help='100 dimension')
parser.add_argument('--lr',type=float,default=0.001,help='learning rate')
parser.add_argument('--data_path',default='../../../../mofan/02/mnist',help='The Path of Generating Data Set Preservation')
parser.add_argument('--save_path',default='gan_s',help='The Path of Generating Data Set Preservation')
opt=parser.parse_args()

# 2.是否使用gpu
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 3.创建保存的路径
if not os.path.exists(opt.save_path):
    os.mkdir(opt.save_path)

# 4.图片预处理
transforms=torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
])

# 5.读入数据
# mnist
dataset=torchvision.datasets.MNIST(opt.data_path,transform=transforms)

dataloader=torch.utils.data.DataLoader(
    dataset=dataset,
    batch_size=opt.batch_size,
    shuffle=True,
    drop_last=True,
    num_workers=2,
)


# 6.框架
class MyGanD(torch.nn.Module):
    '''鉴别器'''
    def __init__(self):
        super(MyGanD,self).__init__()
        self.fc=torch.nn.Sequential(
            torch.nn.Linear(28*28,16*16),
            torch.nn.LeakyReLU(0.2, True),
            torch.nn.Linear(16 * 16, 1),
            torch.nn.Sigmoid()
        )

    def forward(self, inputs):
        # print(inputs.size())   # [200,1,28,28]
        out=inputs.view(opt.batch_size,-1)  # torch.Size([200, 784])
        out = self.fc(out)
        return out

class MyGanG(torch.nn.Module):
    '''生成器'''
    def __init__(self):
        super(MyGanG,self).__init__()
        self.fc=torch.nn.Sequential(
            torch.nn.Linear(100,16*16),
            torch.nn.ReLU(),
            torch.nn.Linear(16*16,28*28),
            torch.nn.Tanh()
        )
    def forward(self, inputs):
        out=self.fc(inputs)
        out = out.view(-1, 1, 28, 28)
        return out


# 7.二分类损失函数  Adam优化器
GanD=MyGanD().to(device)
GanG=MyGanG().to(device)

criterion=torch.nn.BCELoss()
optimizerD=torch.optim.Adam(GanD.parameters(),lr=opt.lr)
optimizerG=torch.optim.Adam(GanG.parameters(),lr=opt.lr)


# 8.label
label=torch.FloatTensor(opt.batch_size)
real_label=1
fake_label=0


# 9.训练
time_start=datetime.now()
for epoch in range(opt.epoch):
    print(epoch)
    print('开始')
    a_time=datetime.now()
    for i, (imgs,_) in enumerate(dataloader):
        # 1.固定生成器G  训练鉴别器D
        optimizerD.zero_grad()

        #   (1)尽可能把真图鉴别为1
        label.data.fill_(real_label)
        label=label.to(device)
        imgs=imgs.to(device)

        output1=GanD(imgs)
        lossD_real=criterion(output1,label)
        lossD_real.backward()

        #   (2)尽可能把假图鉴别为0
        label.data.fill_(fake_label)
        label=label.to(device)

        noise=torch.randn(opt.batch_size,opt.nz)
        fake=GanG(noise)  # 得到假图

        output2=GanD(fake.detach())  # 避免梯度传到G网络
        lossD_fake=criterion(output2,label)
        lossD_fake.backward()
        lossD=lossD_real+lossD_fake
        optimizerD.step()

        # 2.固定鉴别器D  训练生成器G
        optimizerG.zero_grad()

        label.data.fill_(real_label)
        label=label.to(device)
        output3=GanD(fake)
        lossG=criterion(output3,label)
        lossG.backward()
        optimizerG.step()

        print('Epoch [{}/{}] [{}/{}] d_loss: {:.6f} g_loss: {:.6f} '.format(
            epoch,opt.epoch,i,len(dataloader),lossD,lossG
        ))

        if not os.path.exists('{}/{}/'.format(opt.save_path,str(epoch))):
            os.mkdir('{}/{}/'.format(opt.save_path,str(epoch)))

        # 保图片
        for j in range(opt.batch_size):
            torchvision.utils.save_image(fake.data[j],'{}/{}/fake_epoch{}_{}.png'.format(
                opt.save_path, str(epoch), str(epoch), str(j)
            ))

    # 保存每epoch 的完整大图
    if not os.path.exists('{}/all/'.format(opt.save_path)):
        os.mkdir('{}/all/'.format(opt.save_path))

    torchvision.utils.save_image(fake,'{}/all/fake_{}.png'.format(opt.save_path,epoch))


    b_time=datetime.now()
    c_time=b_time-a_time
    print('第{}次 epoch花费时间为{} 平均每批次时间为{}'.format(epoch,c_time,c_time/len(dataloader)))

    # 10.模型保存
    torch.save(GanD.state_dict(),'{}/{}/GanD_{}.pth'.format(opt.save_path,str(epoch),str(epoch)))
    torch.save(GanG.state_dict(),'{}/{}/GanG_{}.pth'.format(opt.save_path,str(epoch),str(epoch)))

time_end=datetime.now()
all_time=time_end-time_start
print()
print('总{}个epoch 共花费时间为{} 平均每次epoch花费时间为{} 平均每批次时间为{}'.format(
    opt.epoch,all_time,all_time/opt.epoch,all_time/(opt.epoch*len(dataloader))
))









