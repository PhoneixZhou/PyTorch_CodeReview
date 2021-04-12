import torch
from torch.utils.data.dataset import TensorDataset
from torch.utils.data.dataloader import DataLoader
#自定义一个类，该类不能被PyTorch原生的pin_memory方法所支持
class SimpleCustomBatch:
    def __init__(self, data):
        transposed_data = list(zip(*data))
        self.inp = torch.stack(transposed_data[0],0)
        self.tgt = torch.stack(transposed_data[1],0)

    def pin_memory(self):
        self.inp = self.inp.pin_memory()
        self.tgt = self.tgt.pin_memory()
        return self

def collate_wrapper(batch):
    return SimpleCustomBatch(batch)

inps = torch.arange(10 * 5, dtype = torch.float32).view(10, 5)
tgts = torch.arange(10 * 5, dtype = torch.float32).view(10, 5)
dataset = TensorDataset(inps, tgts)

loader = DataLoader(dataset, batch_size = 2, collate_fn = collate_wrapper, pin_memory=True)

for batch_ndx, sample in enumerate(loader):
    print(sample.inp.is_pinned())
    print(sample.tgt.is_pinned())
