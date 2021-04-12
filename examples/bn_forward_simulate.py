import torch 
import torch.nn as nn 
import torch.nn.modules.batchnorm 

def create_inputs():
    return torch.randn(8, 3, 20, 20)

#1模拟 BN forward
def dummy_bn_forward(x, bn_weight, bn_bias, eps, mean_val=None, var_val=None):
    if mean_val is None:
        mean_val = x.mean([0, 2, 3])
    if var_val is None:
        var_val = x.var([0, 2, 3], unbiased=False)
    
    x = x - mean_val[None, ..., None, None]
    x = x / torch.sqrt(var_val[None, ..., None, None] + eps)
    x = x * bn_weight[..., None, None] + bn_bias[..., None, None]
    return mean_val, var_val, x


bn_layer = nn.BatchNorm2d(num_features=3)
inputs = create_inputs()
bn_outputs = bn_layer(inputs)
_,_, expected_outputs = dummy_bn_forward(inputs, bn_layer.weight, bn_layer.bias, bn_layer.eps)

assert torch.allclose(expected_outputs, bn_outputs)

#2running_mean、running_var 的更新
running_mean = torch.zeros(3)
running_var = torch.ones_like(running_mean)
momentum = 0.1
bn_layer = nn.BatchNorm2d(num_features=3, momentum=momentum)

#forward 10 times
for t in range(10):
    inputs = create_inputs()
    bn_outputs = bn_layer(inputs)
    inputs_mean, inputs_var,_=dummy_bn_forward(inputs, bn_layer.weight, bn_layer.bias, bn_layer.eps)

    n = inputs.numel()/inputs.size(1)
    running_var = running_var * (1 - momentum) + momentum * inputs_var * n / (n - 1)
    running_mean = running_mean * (1 - momentum) + momentum * inputs_mean

assert torch.allclose(running_var, bn_layer.running_var)
assert torch.allclose(running_mean, bn_layer.running_mean)
print(f'bn_layer running_mean is {bn_layer.running_mean}')
print(f'dummy bn running_mean is {running_mean}')
print(f'bn_layer running_var is {bn_layer.running_var}')
print(f'dummy bn running_var is {running_var}')

#3.
running_mean = torch.zeros(3)
running_var = torch.ones_like(running_mean)
num_batches_tracked = 0
bn_layer = nn.BatchNorm2d(num_features=3, momentum=None)

for t in range(10):
    inputs = create_inputs()
    bn_outputs = bn_layer(inputs)
    inputs_mean, inputs_var,_ = dummy_bn_forward(
        inputs, bn_layer.weight, bn_layer.bias,bn_layer.eps
    )

    num_batches_tracked += 1
    eaf = 1.0 / num_batches_tracked
    n = inputs.numel() / inputs.size(1)
    running_var = running_var * (1 - eaf) + eaf * inputs_var * n / (n - 1)
    running_mean = running_mean * (1 - eaf) + eaf * inputs_mean

assert torch.allclose(running_var, bn_layer.running_var)
assert torch.allclose(running_mean, bn_layer.running_mean)

bn_layer.train(mode=False)
inference_inputs = create_inputs()
bn_outputs = bn_layer(inference_inputs)

_,_, dummy_outputs = dummy_bn_forward(
    inference_inputs, bn_layer.weight,
    bn_layer.bias, bn_layer.eps,
    running_mean, running_var)

assert torch.allclose(dummy_outputs, bn_outputs)
print(f'bn_layer running_mean is {bn_layer.running_mean}')
print(f'dummy bn running_mean is {running_mean}')
print(f'bn_layer running_var is {bn_layer.running_var}')
print(f'dummy bn running_var is {running_var}')

#gamma, beta 的更新

import torchvision
from torchvision.transforms import Normalize, ToTensor, Compose
import torch.nn.functional as F 
from torch.utils.data.dataloader import DataLoader 

mnist = torchvision.datasets.MNIST(root='mnist', download=True, transform=ToTensor())
dataloader = DataLoader(dataset=mnist, batch_size = 8)

#初始化一个带BN的简单模型
toy_model = nn.Sequential(nn.Linear(28 ** 2, 128), nn.BatchNorm1d(128),
                          nn.ReLU(), nn.Linear(128, 10), nn.Sigmoid())
optimizer = torch.optim.SGD(toy_model.parameters(), lr = 0.1)

bn_1d_layer = toy_model[1]

print(f'Initial weight is {bn_layer.weight[:4].tolist()}...')
print(f'Initial bias is {bn_layer.bias[:4].tolist()}...\n')

for (i, data) in enumerate(dataloader):
    output = toy_model(data[0].view(data[0].shape[0], -1))
    (F.cross_entropy(output, data[1])).backward()

    print(f'Gradient of weight is {bn_1d_layer.weight.grad[:4].tolist()}...')
    print(f'Gradient of bias is {bn_1d_layer.bias.grad[:4].tolist()}...')

    optimizer.step()
    optimizer.zero_grad()
    if i == 1:
        break

print(f'\nNow weight is {bn_1d_layer.weight[:4].tolist()}...')
print(f'Now bias is {bn_1d_layer.bias[:4].tolist()}...')

inputs = torch.randn(4, 128)
bn_outputs = bn_1d_layer(inputs)
new_bn = nn.BatchNorm1d(128)
bn_outputs_no_weight_bias = new_bn(inputs)

assert not torch.allclose(bn_outputs, bn_outputs_no_weight_bias)




