'''
模型的保存

'''

import torch

x=torch.Tensor([[3,1],[0,2]])
y=torch.Tensor([[1],[2]])

class MyNet(torch.nn.Module):
    def __init__(self):
        super(MyNet,self).__init__()
        self.fc=torch.nn.Linear(2,1)
        self.opt=torch.optim.Adam(self.parameters(),lr=0.01)
        self.lms=torch.nn.MSELoss()

    def forward(self, inputs):
        return self.fc(inputs)


model=MyNet()
for i in range(1000):
    out=model.forward(x)
    loss=model.lms(out,y)
    print(loss)
    model.opt.zero_grad()
    loss.backward()
    model.opt.step()

torch.save(model.state_dict(),'te.t7')
print(out)

