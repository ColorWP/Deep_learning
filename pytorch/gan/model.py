'''
对抗神经网络gan

这里我没有使用池化操作
原因是池化操作可能会丢失一些比较重要的数据

众所周知了  池化会减小图片尺寸  一般有2中池化操作：
平均池化：计算图像区域的平均值作为该区域池化后的值。  AvgPool2d(2, stride=2)
最大池化：选图像区域的最大值作为该区域池化后的值。    MaxPool2d(kernel_size=2)

当然 也可能会过拟合  所以如果有充足时间 可以2种方法都试试  一种加池化，一种不加池化  看看效果
'''

import torch.nn as nn
# 定义生成器网络G
class NetG(nn.Module):
    def __init__(self, ngf, nz):
        super(NetG, self).__init__()
        # layer1输入的是一个100x1x1的随机噪声, 输出尺寸(ngf*8)x4x4
        self.layer1 = nn.Sequential(
            nn.ConvTranspose2d(nz, ngf * 8, kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(ngf * 8), # 归一化
            nn.ReLU(inplace=True)
        )
        # layer2输出尺寸(ngf*4)x8x8
        self.layer2 = nn.Sequential(
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(inplace=True)
        )
        # layer3输出尺寸(ngf*2)x16x16
        self.layer3 = nn.Sequential(
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(inplace=True)
        )
        # layer4输出尺寸(ngf)x32x32
        self.layer4 = nn.Sequential(
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(inplace=True)
        )
        # layer5输出尺寸 3x96x96
        self.layer5 = nn.Sequential(
            nn.ConvTranspose2d(ngf, 3, 5, 3, 1, bias=False),
            nn.Tanh()
        )

    # 定义NetG的前向传播
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        return out


# 定义鉴别器网络D
class NetD(nn.Module):
    def __init__(self, ndf):
        super(NetD, self).__init__()
        # layer1 输入 3 x 96 x 96, 输出 (ndf) x 32 x 32
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, ndf, kernel_size=5, stride=3, padding=1, bias=False),
            nn.BatchNorm2d(ndf),
            nn.LeakyReLU(0.2, inplace=True)
        )
        # layer2 输出 (ndf*2) x 16 x 16
        self.layer2 = nn.Sequential(
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True)
        )
        # layer3 输出 (ndf*4) x 8 x 8
        self.layer3 = nn.Sequential(
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True)
        )
        # layer4 输出 (ndf*8) x 4 x 4
        self.layer4 = nn.Sequential(
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True)
        )
        # layer5 torch.Size([200, 1, 1, 1])   输出一个数(概率) torch.Size([200, 1, 1, 1])
        self.layer5 = nn.Sequential(
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )


    # 定义NetD的前向传播
    def forward(self,x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        # print(out)  # 查看数据是否会变

        # 这里将torch.Size([200, 1, 1, 1])  转化为  torch.Size([200])
        # 与后面的label 进行二分类交叉熵损失
        out=out.view(out.size(0))
        # print(out)  # 查看数据是否会改变
        # print(out.size())

        return out

