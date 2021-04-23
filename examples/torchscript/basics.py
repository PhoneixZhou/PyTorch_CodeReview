import torch  # This is all you need to use both PyTorch and TorchScript!
print(torch.__version__)

class MyDecisionGate(torch.nn.Module):
    def forward(self, x):
        if x.sum() > 0:
            return x
        else:
            return -x

class MyCell(torch.nn.Module):
    def __init__(self, dg):
        super(MyCell, self).__init__()
        #self.dg = MyDecisionGate()
        self.dg = dg
        self.linear = torch.nn.Linear(4, 4)

    def forward(self, x, h):
        new_h = torch.tanh(self.dg(self.linear(x)) + h)
        #new_h = torch.tanh(self.linear(x) + h)
        return new_h, new_h

# my_cell = MyCell()
# x = torch.rand(3, 4)
# h = torch.rand(3, 4)

# print(my_cell)
# print(my_cell(x, h))

#PyTorch record operations as they occur, and replay them backwards in computing
# derivatives. In this way, the framework does not have to explicitly define derivatives
# for all constructs in the language.

# my_cell_tracing = MyCell()
# x, h = torch.rand(3, 4), torch.rand(3, 4)
# traced_cell = torch.jit.trace(my_cell_tracing, (x, h))

# print(traced_cell)
# print(traced_cell.graph)
# print(traced_cell.dg.code)
# print(traced_cell.code)
#traced_cell(x, h)

# scripted_gate = torch.jit.script(MyDecisionGate())
# my_cell_scripted = MyCell()
# scripted_cell = torch.jit.script(my_cell_scripted)

# print(scripted_gate.code)
# print(scripted_cell.code)


print("---------------------------------------------------------")

x, h = torch.rand(3, 4), torch.rand(3, 4)

scripted_gate = torch.jit.script(MyDecisionGate())
class MyRNNLoop(torch.nn.Module):
    def __init__(self):
        super(MyRNNLoop, self).__init__()
        self.cell = torch.jit.trace(MyCell(scripted_gate), (x, h))
    
    def forward(self, xs):
        h, y = torch.zeros(3, 4), torch.zeros(3, 4)
        for i in range(xs.size(0)):
            y, h = self.cell(xs[i], h)
        return y, h

rnn_loop = torch.jit.script(MyRNNLoop())
print(rnn_loop.code)
print(rnn_loop.cell.code)
print(rnn_loop.cell.dg.code)

print("-----------------------------------------------")

class WrapRNN(torch.nn.Module):
    def __init__(self):
        super(WrapRNN, self).__init__()
        self.loop = torch.jit.script(MyRNNLoop())
    
    def forward(self, xs):
        y, h = self.loop(xs)
        return torch.relu(y)
    
traced = torch.jit.trace(WrapRNN(), (torch.rand(10, 3, 4)))
print(traced.code)

print("-------------------------------------------------------")
traced.save("wrapped_rnn.pt")
loaded = torch.jit.load("wrapped_rnn.pt")

print(loaded)
print(loaded.code)
    

    




