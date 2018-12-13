'''
测试  生成对抗神经网络gan
mnist 28*28
全连接
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
parser.add_argument('--save_path',default='testS',help='The Path of Generating Data Set Preservation')
opt=parser.parse_args()

# 2.是否使用gpu
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 3.创建保存的路径
if not os.path.exists(opt.save_path):
    os.mkdir(opt.save_path)

# 4.框架
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


# 5.加载模型
GanD=MyGanD().to(device)
GanG=MyGanG().to(device)

GanD.load_state_dict(torch.load('./moxingS/GanD_99.pth'))
GanG.load_state_dict(torch.load('./moxingS/GanG_99.pth'))


# 6.测试
# 想要生成多少张图片 只需要修改 batch_size 的大小即可

noise = torch.randn(opt.batch_size, opt.nz)
fake = GanG(noise)  # 得到假图

# batch_size=200
for j in range(opt.batch_size):
    torchvision.utils.save_image(fake.data[j],
                                 '{}/gan_test{}.png'.format(opt.save_path,str(j)),
                                 normalize=True)








