
import torch

class MyCnn(torch.nn.Module):
    def __init__(self):
        super(MyCnn,self).__init__()
        # 1,28,28    16,28,28   16,14,14
        self.conv1=torch.nn.Sequential(
            torch.nn.Conv2d(1,16,5,1,2),
            # torch.nn.Dropout(0.5),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2)
        )
        # 16,14,14  32,14,14    32,7,7
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(16, 32, 5, 1, 2),
            # torch.nn.Dropout(0.5),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2)
        )
        # 32,7,7
        self.out1=torch.nn.Linear(32*7*7,10)

    def forward(self, inputs):
        x=self.conv1(inputs)
        x=self.conv2(x)
        x=x.view(x.size(0),-1)  # 展平多维的卷积图成 (batch_size, 32 * 7 * 7)
        output=self.out1(x)
        return output




