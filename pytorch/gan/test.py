import argparse
import torch
import torchvision
import torchvision.utils as vutils
import torch.nn as nn
from random import randint
from model import NetD, NetG
import os
from datetime import datetime

a=os.path.exists('data/')
print(a)

parser = argparse.ArgumentParser()
parser.add_argument('--batchSize', type=int, default=1)
parser.add_argument('--imageSize', type=int, default=96)
parser.add_argument('--nz', type=int, default=100, help='size of the latent z vector')
parser.add_argument('--ngf', type=int, default=64)
parser.add_argument('--ndf', type=int, default=64)
parser.add_argument('--epoch', type=int, default=20, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.0002, help='learning rate, default=0.0002')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--outf', default='test/', help='folder to output images and model checkpoints')
opt = parser.parse_args()

# 定义是否使用GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


netG = NetG(opt.ngf, opt.nz).to(device)
netD = NetD(opt.ndf).to(device)


netG.load_state_dict(torch.load('./moxing/netG_20.pth'))
netD.load_state_dict(torch.load('./moxing/netD_20.pth'))

img_num=300   # 生成多少图片数据
# 这里生成图片
for j in range(img_num):
    noise = torch.randn(opt.batchSize, opt.nz, 1, 1)
    fake = netG(noise)
    for i in range(opt.batchSize):
        vutils.save_image(fake.data[i],
                          '{}fake_samples_epoch_{}.png'.format(opt.outf, str(j)),
                          normalize=True)


