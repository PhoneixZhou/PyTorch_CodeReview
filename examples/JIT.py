import torch
import torchvision.models as models
#1.trace
resnet = torch.jit.trace(models.resnet18(), torch.rand(1, 3, 224, 224))

output =resnet(torch.ones(1,3,224,224))
print(output)

output = resnet(torch.ones(1,3,224,224))
resnet.save('resnet.pt')

#2.trace & script
#1 大部分情况 model 只有 tensor operation，就直接无脑 tracing 
#2 带 control-flow (if-else, for-loop) 的，上 scripting 
#3 碰上 scripting 不能 handle 的语法，要么重写，要么把 tracing 和 scripting 合起来用
# （比如说只在有 control-flow 的代码用 scripting，其他用 tracing）
import torch 
import torch.nn as nn 
import torch.nn.functional as F 

class MyModule(nn.Module):
    def __init__(self):
        super(MyModule, self).__init__()
        #torch.jit.trace produces a ScriptModule's conv1 and conv2
        self.conv1 = torch.jit.trace(nn.Conv2d(1, 20, 5), torch.rand(1, 1, 16, 16))
        self.conv2 = torch.jit.trace(nn.Conv2d(20, 20, 5), torch.rand(1, 20, 16, 16))

    def forward(self, input):
        input = F.relu(self.conv1(input))
        input = F.relu(self.conv2(input))
        return input 

scripted_module = torch.jit.script(MyModule())


