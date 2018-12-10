'''

1.固定生成器G，训练鉴别器D，optimizerD.zero_grad()
    1.来一张真图片 传到D网络得到output 并进行label=1操作，之后进行二分类交叉熵得到lossD_real，再进行反向传播
    2.把label=0，随机生成数据，并用G网络 生成一张假图，通过D网络 得到output 此时需要netD(fake.detach())
        2.1 意思是  当反向传播经过这个node时，梯度就不会从这个node往前面传播
        2.2 简单说  就是之后反向传播时，不会对G网络进行训练，只优化D网络
    3.在进行二分类交叉熵得到lossD_fake 再进行反向传播 之后lossD相加 进行优化

2.固定鉴别器D，训练生成器G，optimizerG.zero_grad()
    1.让D尽可能把G生成的假图判别为1
    2.先让label=1,将假图传入D网络进行鉴别得到output 再进行二分类交叉熵 得到G的lossG 之后反向传播 优化


drop_last 不够一个batch时 是否丢弃数据

output = netD(fake.detach()) 中的  detach()
简单来说detach就是截断反向传播的梯度流

我们看源码 或者pycharm中ctrl + 鼠标左键 点击detach() 来看它的注释

detach = _add_docstr(_C._TensorBase.detach, r"""
    Returns a new Tensor, detached from the current graph.

    The result will never require gradient.

    .. note::

      Returned Tensor uses the same data tensor as the original one.
      In-place modifications on either of them will be seen, and may trigger
      errors in correctness checks.
    """)

原理：就是 将某个node变成不需要梯度的Varibale。因此当反向传播经过这个node时，梯度就不会从这个node往前面传播
解释：GAN的G的更新，主要是GAN loss。就是G生成的fake图让D来判别，得到的损失，计算梯度进行反传。这个梯度只能影响G，不能影响D

cpu: 总共花费时间为18:39:54.992707 平均每个epoch花费时间为0:46:39.791363


'''

import argparse
import torch
import torchvision
import torchvision.utils as vutils
import torch.nn as nn
from model import NetD, NetG
import os
from datetime import datetime

a=os.path.exists('/media/colorwp/空间/01wp/d-python/cx/content/pytorch/01/')
print(a)

parser = argparse.ArgumentParser()
parser.add_argument('--batchSize', type=int, default=200)
parser.add_argument('--imageSize', type=int, default=96)
parser.add_argument('--nz', type=int, default=100, help='size of the latent z vector')
parser.add_argument('--ngf', type=int, default=64)
parser.add_argument('--ndf', type=int, default=64)
parser.add_argument('--epoch', type=int, default=24, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.0002, help='learning rate, default=0.0002')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--data_path', default='/media/colorwp/空间/01wp/d-python/cx/content/pytorch/01/', help='folder to train data')
parser.add_argument('--outf', default='imgs/', help='folder to output images and model checkpoints')
opt = parser.parse_args()
# 定义是否使用GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#图像读入与预处理
transforms = torchvision.transforms.Compose([
    torchvision.transforms.Resize(opt.imageSize),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), ])

dataset = torchvision.datasets.ImageFolder(opt.data_path, transform=transforms)

dataloader = torch.utils.data.DataLoader(
    dataset=dataset,
    batch_size=opt.batchSize,
    shuffle=True,
    drop_last=True,
)

netG = NetG(opt.ngf, opt.nz).to(device)
netD = NetD(opt.ndf).to(device)

criterion = nn.BCELoss()   # 二分类交叉熵
optimizerG = torch.optim.Adam(netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
optimizerD = torch.optim.Adam(netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

label = torch.FloatTensor(opt.batchSize)   # torch.Size([200])
real_label = 1
fake_label = 0

time_start=datetime.now()
for epoch in range(1, opt.epoch + 1):
    print(epoch)
    print('开始')
    a_time=datetime.now()
    for i, (imgs,_) in enumerate(dataloader):
        # print(imgs.data.size())
        # 固定生成器G，训练鉴别器D
        optimizerD.zero_grad()
        ## 让D尽可能的把真图片判别为1
        imgs=imgs.to(device)
        print('开始调用D')
        output = netD(imgs)
        label.data.fill_(real_label)  # 填充
        label=label.to(device)
        errD_real = criterion(output, label)
        errD_real.backward()
        ## 让D尽可能把假图片判别为0
        label.data.fill_(fake_label)
        ## 随机生成数据
        noise = torch.randn(opt.batchSize, opt.nz, 1, 1)
        noise=noise.to(device)
        fake = netG(noise)  # 生成假图
        output = netD(fake.detach()) # 避免梯度传到G，因为G不用更新
        errD_fake = criterion(output, label)
        errD_fake.backward()
        errD = errD_fake + errD_real
        optimizerD.step()

        # 固定鉴别器D，训练生成器G
        optimizerG.zero_grad()
        # 让D尽可能把G生成的假图判别为1
        label.data.fill_(real_label)
        label = label.to(device)
        output = netD(fake)
        errG = criterion(output, label)
        errG.backward()
        optimizerG.step()

        print('[%d/%d][%d/%d] Loss_D: %.3f Loss_G %.3f'
              % (epoch, opt.epoch, i, len(dataloader), errD.item(), errG.item()))

        if not os.path.exists('{}{}/'.format(opt.outf, str(epoch))):
            os.mkdir('{}{}/'.format(opt.outf, str(epoch)))

        for i in range(opt.batchSize):
            vutils.save_image(fake.data[i],
                              '{}{}/fake_samples_epoch_{}.png'.format(opt.outf, str(epoch),str(i)),
                              normalize=True)
    b_time = datetime.now()
    c_time=b_time-a_time
    print('第{}次 执行时间为{} 平均每次时间为{}'.format(epoch,c_time,c_time/len(dataloader)))
    torch.save(netG.state_dict(), r'{}{}/netG_{}.pth'.format(opt.outf, str(epoch),epoch))
    torch.save(netD.state_dict(), r'{}{}/netD_{}.pth'.format(opt.outf, str(epoch),epoch))
time_end=datetime.now()
all_time=time_end-time_start
print()
print('总共花费时间为{} 平均每个epoch花费时间为{}'.format(all_time,all_time/opt.epoch))
