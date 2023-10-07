import torch.nn
from torch._dynamo.backends.common import aot_autograd
from functorch.compile import make_boxed_func


class MyModule(torch.nn.Module):
    def __init__(self, device: torch.device):
        super().__init__()
        self.layer1 = torch.nn.Linear(10, 1024, True, device)
        self.activate_fn = torch.nn.ReLU()
        self.layer2 = torch.nn.Linear(1024, 10, True, device)
        self.loss = torch.nn.CrossEntropyLoss().to(device)
        self.optim = torch.optim.Adam(self.parameters())

    def forward(self, t: torch.Tensor, l: torch.Tensor) -> torch.Tensor:
        self.optim.zero_grad()
        output = self.layer1(t)
        output = self.activate_fn(output)
        output = self.layer2(output)
        loss = self.loss(torch.nn.functional.softmax(output, -1), l)
        loss.backward()
        self.optim.step()


def compile_fn(gm: torch.fx.GraphModule, inputs):
    gm.graph.print_tabular()
    return make_boxed_func(gm.forward)


if __name__ == '__main__':
    device = torch.device("cuda:0")
    model = MyModule(device)
    compile_model = torch.compile(model, backend=aot_autograd(fw_compiler=compile_fn, bw_compiler=compile_fn))
    compile_model(torch.randn([2, 10]).to(device), torch.randint(1, (2,)).to(device))
