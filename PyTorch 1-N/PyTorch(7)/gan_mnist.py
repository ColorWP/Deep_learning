'''
生成对抗神经网络gan
mnist 数据尺寸  28*28
鉴别器：# (batch,1,28,28) --> (batch,(64*1),14,14) (batch,(64*2),7,7) (batch,64*4,3,3)  (batch,1,1,1)
生成器：# (batch,100,56*56) --> (batch,(64*1),56,56) (batch,(64*2),56,56) (batch,64*4,4,4)  (batch,1,1,1)

第2次 执行时间为0:38:04.125075 平均每次时间为0:00:07.613750
'''


import argparse
import torch
import torch.utils.data
import torchvision
import os

from datetime import datetime


# 1.定义超参数
parser=argparse.ArgumentParser()
parser.add_argument('--batch_size',type=int,default=200,help='Batch processing')
# parser.add_argument('--img_resize',type=int,default=96,help='Resize image')
parser.add_argument('--lr',type=float,default=0.001,help='learning rate')
parser.add_argument('--epoch', type=int, default=100, help='All data trained several times')
parser.add_argument('--ndf', type=int, default=64, help='convolution')
parser.add_argument('--nz', type=int, default=100, help='Random size')
parser.add_argument('--ngf', type=int, default=56*56, help='deconvolution')
parser.add_argument('--data_path',default='../../../../mofan/02/mnist',help='Path of Training Data Set')
parser.add_argument('--save_path',default='gan_c',help='The Path of Generating Data Set Preservation')
opt=parser.parse_args()

# 2.定义是否使用gpu
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

if not os.path.exists(opt.save_path):
    os.mkdir(opt.save_path)

# 3.图片预处理
transforms=torchvision.transforms.Compose([
    # torchvision.transforms.Resize(opt.img_resize),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
])


# 4.读入数据
# 如果是个人搜集的图片数据
# dataset = torchvision.datasets.ImageFolder(opt.data_path, transform=transforms)

# mnist
dataset=torchvision.datasets.MNIST(opt.data_path,transform=transforms)

dataloader=torch.utils.data.DataLoader(
    dataset=dataset,
    batch_size=opt.batch_size,
    shuffle=True,
    drop_last=True,
    num_workers=2,
)

def to_img(x):
    # print(111)
    # print(x.size())
    out = 0.5 * (x + 1)
    # print(out.size())
    out = out.clamp(0, 1)
    # print(out.size())
    out = out.view(-1, 1, 28, 28)
    # print(out.size())
    return out



# 5.框架
class MyGan_D(torch.nn.Module):
    '''鉴别器'''
    # (batch,1,28,28) --> (batch,(64*1),14,14) (batch,(64*2),7,7) (batch,64*4,3,3)  (batch,1,1,1)
    def __init__(self,ndf):
        super(MyGan_D,self).__init__()
        self.conv1=torch.nn.Sequential(
            torch.nn.Conv2d(1,ndf,kernel_size=5,stride=1,padding=2,bias=False),
            torch.nn.BatchNorm2d(ndf),
            torch.nn.LeakyReLU(0.2,inplace=True),
            torch.nn.AvgPool2d(2, stride=2),  # batch, 64, 14, 14
        )
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(ndf, ndf*2, kernel_size=5, stride=1, padding=2, bias=False),
            torch.nn.BatchNorm2d(ndf*2),
            torch.nn.LeakyReLU(0.2, inplace=True),
            torch.nn.AvgPool2d(2, stride=2),  # batch, 64*2, 7, 7
        )
        self.conv3 = torch.nn.Sequential(
            torch.nn.Conv2d(ndf*2, ndf * 4, kernel_size=5, stride=1, padding=2, bias=False),
            torch.nn.BatchNorm2d(ndf * 4),
            torch.nn.LeakyReLU(0.2, inplace=True),
            torch.nn.AvgPool2d(2, stride=2),  # batch, 64*4, 3, 3
        )
        self.conv4 = torch.nn.Sequential(
            torch.nn.Conv2d(ndf * 4, 1, kernel_size=3, stride=1, padding=0, bias=False),
            torch.nn.Sigmoid()
        )

    def forward(self,x):
        # print(x.size())     # torch.Size([200, 1, 28, 28])
        out=self.conv1(x)     # torch.Size([200, 64, 14, 14])
        out=self.conv2(out)   # torch.Size([200, 64*2, 7, 7])
        out=self.conv3(out)   # torch.Size([200, 64*4, 4, 4])
        out=self.conv4(out)   # torch.Size([200, 1, 1, 1])
        out=out.view(out.size(0))   # torch.Size([200])
        # print(out.size())
        return out

class MyGan_G(torch.nn.Module):
    '''生成器'''
    # (batch,100,56*56) --> (batch,(64*1),56,56) (batch,(64*2),56,56) (batch,64*4,4,4)  (batch,1,1,1)

    def __init__(self,nz,ngf):
        super(MyGan_G,self).__init__()
        self.fc=torch.nn.Linear(nz,ngf)   # torch.Size([200, 100]) -->[200,3136]
        self.d_conv1=torch.nn.Sequential(
            torch.nn.Conv2d(1,64,3,stride=1,padding=1),  # (batch,(64*1),56,56)
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(True)
        )
        self.d_conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(64, 32, 3, stride=1, padding=1),  # (batch,(32),56,56)
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(True)
        )
        self.d_conv3 = torch.nn.Sequential(
            torch.nn.Conv2d(32, 1, 2, stride=2),  # (batch,1,28,28)
            torch.nn.Tanh()
        )


    def forward(self,x):
        out=self.fc(x)   # torch.Size([200, 3136])
        out=out.view(out.size(0),1,56,56)  # torch.Size([200, 1, 56, 56])
        out=self.d_conv1(out)
        out=self.d_conv2(out)
        out=self.d_conv3(out)
        return out



# 6.二分类交叉熵损失函数 和 Adam优化器
GanD = MyGan_D(opt.ndf).to(device)
GanG = MyGan_G(opt.nz,opt.ngf).to(device)

criterion=torch.nn.BCELoss()
optimizerD=torch.optim.Adam(GanD.parameters(),lr=opt.lr)
optimizerG=torch.optim.Adam(GanG.parameters(),lr=opt.lr)


# 7.label
label=torch.FloatTensor(opt.batch_size)
real_label=1
fake_label=0


# 8.开始
print(len(dataloader))  # 300
time_start=datetime.now()
for epoch in range(1,opt.epoch+1):
    print(epoch)
    print('开始')
    a_time=datetime.now()
    for i, (imgs,_) in enumerate(dataloader):
        # 1.固定生成器，训练判别器
        optimizerD.zero_grad()
        #   (1)让D尽可能把真图判别为1
        imgs=imgs.to(device)
        label.data.fill_(real_label)
        label=label.to(device)

        output1=GanD(imgs)
        lossD_real=criterion(output1,label)
        lossD_real.backward()

        #   (2)让D尽可能把假图判别为0
        label.data.fill_(fake_label)
        label=label.to(device)

        noise=torch.randn(opt.batch_size,opt.nz)  # 随机生成数据
        fake=GanG(noise)  # 生成假图

        output2=GanD(fake.detach()) # 避免梯度传到G
        lossD_fake=criterion(output2,label)
        lossD_fake.backward()

        lossD=lossD_real+lossD_fake  # D网络的损失
        optimizerD.step()  # D网络优化

        # 2.固定鉴别器D，训练生成器G
        optimizerG.zero_grad()
        # 让D尽可能把G生成的假图判别为1
        label.data.fill_(real_label)
        label=label.to(device)

        output3=GanD(fake)
        lossG=criterion(output3,label)
        lossG.backward()
        optimizerG.step()

        print('Epoch [{}/{}] [{}/{}],d_loss: {:.6f},g_loss: {:.6f}'.format(epoch,opt.epoch,i,len(dataloader),lossD,lossG))

        if not os.path.exists('{}/{}/'.format(opt.save_path, str(epoch))):
            os.mkdir('{}/{}/'.format(opt.save_path, str(epoch)))

        for j in range(opt.batch_size):  # to_img(fake.data[j])
            torchvision.utils.save_image(fake.data[j],'{}/{}/fake_epoch{}_{}.png'.format(
                opt.save_path, str(epoch), str(epoch), str(j)
            ))

    # 保存每epoch 的完整大图
    if not os.path.exists('{}/all/'.format(opt.save_path)):
        os.mkdir('{}/all/'.format(opt.save_path))

    torchvision.utils.save_image(fake, '{}/all/fake_{}.png'.format(opt.save_path, epoch))

    b_time=datetime.now()
    c_time=b_time-a_time
    print('第{}次 执行时间为{} 平均每次时间为{}'.format(str(epoch),c_time,c_time/len(dataloader)))

    # 9.模型保存
    torch.save(GanD.state_dict(),'{}/{}/GanD_{}.pth'.format(opt.save_path,str(epoch),str(epoch)))
    torch.save(GanG.state_dict(),'{}/{}/GanG_{}.pth'.format(opt.save_path,str(epoch),str(epoch)))

time_end=datetime.now()
all_time=time_end-time_start
print()
print('总{}个epoch 共花费时间为{} 平均每次花费时间为{}'.format(str(opt.epoch),all_time,all_time/opt.epoch))


