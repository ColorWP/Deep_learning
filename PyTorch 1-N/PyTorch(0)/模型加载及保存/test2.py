'''
模型的加载
'''

import torch

x=torch.Tensor([[3,1],[0,2]])
# y=torch.Tensor([[1],[0]])

class MyNet(torch.nn.Module):
    def __init__(self):
        super(MyNet,self).__init__()
        self.fc=torch.nn.Linear(2,1)
        self.opt=torch.optim.Adam(self.parameters(),lr=0.01)
        self.lms=torch.nn.MSELoss()

    def forward(self, inputs):
        return self.fc(inputs)



model=MyNet()
model.load_state_dict(torch.load('te.t7'))

a=model.forward(x)
print(a)

