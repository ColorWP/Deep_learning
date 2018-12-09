'''利用线性回归 了解pytorch'''

import torch
import numpy as np
import matplotlib.pyplot as plt

# 制造数据点
x_train = np.array([[3.3], [4.4], [5.5], [6.71], [6.93], [4.168],
                    [9.779], [6.182], [7.59], [2.167], [7.042],
                    [10.791], [5.313], [7.997], [3.1]], dtype=np.float32)
y_train = np.array([[1.7], [2.76], [2.09], [3.19], [1.694], [1.573],
                    [3.366], [2.596], [2.53], [1.221], [2.827],
                    [3.465], [1.65], [2.904], [1.3]], dtype=np.float32)

# 画图
plt.plot(x_train,y_train,'r.')
plt.show()

# 把numpy 转化为 tensor
x_train = torch.from_numpy(x_train)
y_train = torch.from_numpy(y_train)


# 框架  LinearRegression
class MyNet(torch.nn.Module):
    def __init__(self):
        super(MyNet, self).__init__()
        self.linear = torch.nn.Linear(1, 1)  # 输入和输出是1维
    def forward(self, x):
        out = self.linear(x)
        return out
model = MyNet()

# 定义损失和优化
criterion = torch.nn.MSELoss()  # 均方误差
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)


# 训练
num_epochs = 1000
for epoch in range(num_epochs):
    # forward
    out = model(x_train) # 前向传播
    loss = criterion(out, y_train) # 计算loss

    # backward
    optimizer.zero_grad() # 梯度归零
    loss.backward() # 方向传播
    optimizer.step() # 更新参数

    # 输出 损失
    if (epoch+1) % 20 == 0:
        print('Epoch[{}/{}], loss: {:.6f}'.format(epoch+1,
                                                  num_epochs,
                                                  loss.data.item()))

        plt.plot(x_train.data.numpy(),y_train.data.numpy(),'r.')
        plt.plot(x_train.data.numpy(),out.data.numpy(),'r-')
        plt.show()

# # 测试
# model.eval()  # 让model变成测试模式  这里 可以重新造数据x_train=
# predict = model(x_train)
# predict = predict.data.numpy()
#
# # 查看预测结果
# plt.plot(x_train.data.numpy(),y_train.data.numpy(),'r.')
# plt.plot(x_train.data.numpy(),predict,'r-')
# plt.show()
