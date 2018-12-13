'''
测试  生成对抗神经网络gan
mnist 数据尺寸  28*28
鉴别器：# (batch,1,28,28) --> (batch,(64*1),14,14) (batch,(64*2),7,7) (batch,64*4,3,3)  (batch,1,1,1)
生成器：# (batch,100,56*56) --> (batch,(64*1),56,56) (batch,(64*2),56,56) (batch,64*4,4,4)  (batch,1,1,1)
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
parser.add_argument('--save_path',default='testC',help='The Path of Generating Data Set Preservation')
opt=parser.parse_args()

# 2.定义是否使用gpu
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

if not os.path.exists(opt.save_path):
    os.mkdir(opt.save_path)

# 3.框架
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


# 4.二分类交叉熵损失函数 和 Adam优化器
GanD = MyGan_D(opt.ndf).to(device)
GanG = MyGan_G(opt.nz,opt.ngf).to(device)

GanD.load_state_dict(torch.load('./moxingC/GanD_46.pth'))
GanG.load_state_dict(torch.load('./moxingC/GanG_46.pth'))


# 5.测试生成图片

noise=torch.randn(opt.batch_size,opt.nz)
fake=GanG(noise)
print('开始生成图片')
for j in range(opt.batch_size):
    torchvision.utils.save_image(fake.data[j],'{}/fake_testC_{}.png'.format(
        opt.save_path,str(j)
    ))
print('已成功生成 {}张图片'.format(opt.batch_size))



