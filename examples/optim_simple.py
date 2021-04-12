import torch 
import numpy as np 
import warnings
warnings.filterwarnings('ignore')

x = torch.linspace(-np.pi, np.pi, 2000)
y = torch.sin(x)

p = torch.tensor([1, 2, 3])
xx = x.unsqueeze(-1).pow(p)

model = torch.nn.Sequential(
    torch.nn.Linear(3, 1),
    torch.nn.Flatten(0, 1)
)

loss_fn = torch.nn.MSELoss(reduction='sum')
learning_rate = 1e-3
optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate)

for t in range(1, 1001):
    y_pred = model(xx)
    loss = loss_fn(y_pred, y)
    if t % 100 == 0:
        print('No.{: 5d}, loss: {:.6f}'.format(t, loss.item()))
    optimizer.zero_grad() # 梯度清零
    loss.backward() # 反向传播计算梯度
    optimizer.step() # 梯度下降法更新参数


#2.
from torch.optim import SGD
from torch import nn

class DummyModel(nn.Module):
    def __init__(self, class_num=10):
        super(DummyModel, self).__init__() 
        self.base = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(128, class_num)

    def forward(self, x):
        x = self.base(x)
        x = self.gap(x)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        return x

model = DummyModel()
optimizer = SGD([
                {'params': model.base.parameters()}, 
                {'params': model.fc.parameters(), 'lr': 1e-3} # 对 fc的参数设置不同的学习率
            ], lr=1e-2, momentum=0.9)

