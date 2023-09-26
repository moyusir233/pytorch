import torch
from torchvision.models.resnet import resnet152
from transformers import AutoTokenizer, AutoModel
from typing import Iterator, Generator, Tuple
import torch.utils.benchmark as benchmark


class CudaEventTimer:
    def __init__(self):
        self.start_event: torch.cuda.Event = torch.cuda.Event(enable_timing=True)
        self.end_event: torch.cuda.Event = torch.cuda.Event(enable_timing=True)
        self.spend_time: int = 0

    def __enter__(self):
        self.start_event.record()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_event.record()
        self.end_event.synchronize()
        self.spend_time = self.start_event.elapsed_time(self.end_event)

    def get_spend_time(self) -> int:
        return self.spend_time


class VisionModel(torch.nn.Module):
    def __init__(self, model: torch.nn.Module, optim: torch.optim.Optimizer,
                 loss_fn: torch.nn.Module, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = model
        self.optim = optim
        self.loss_fn = loss_fn

    def forward(self, data, label):
        self.optim.zero_grad()
        output: torch.Tensor = self.model.forward(data)
        loss: torch.Tensor = self.loss_fn.forward(output, label)
        loss.backward()
        self.optim.step()


def create_vision_dataloader(batch_size: int, class_num: int, device: torch.device) -> Generator[
    Tuple[torch.Tensor, torch.Tensor], None, None]:
    while 1:
        yield (torch.randn(batch_size, 3, 128, 128).to(torch.float32).to(device),
               torch.randint(class_num, (batch_size,)).to(device)
               )


def compile_vision_model():
    device = torch.device("cuda:0")
    model = resnet152().to(device)
    optim = torch.optim.Adam(params=model.parameters())
    loss_fn = torch.nn.CrossEntropyLoss().to(device)
    dataloader = create_vision_dataloader(64, 1000, device)

    vision_model = VisionModel(model, optim, loss_fn)
    compile_model = torch.compile(vision_model, mode="max-autotune")
    # compile_model = vision_model

    # warm up
    for _ in range(5):
        data, label = next(dataloader)
        compile_model(data, label)
        vision_model.forward(data, label)
    torch.cuda.synchronize(device)

    timer = CudaEventTimer()
    for i in range(10):
        data, label = next(dataloader)

        with timer:
            compile_model(data, label)
        print("epoch:{}, compile model spend time:{}ms".format(i + 1, timer.get_spend_time()))

        with timer:
            vision_model.forward(data, label)
        print("epoch:{}, original model spend time:{}ms".format(i + 1, timer.get_spend_time()))


class LanguageModel(torch.nn.Module):
    def __init__(self, tokenizer: AutoTokenizer, model: AutoModel, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tokenizer = tokenizer
        self.model = model

    def forward(self, sentence: str):
        inputs = self.tokenizer(sentence, return_tensors="pt")


if __name__ == '__main__':
    compile_vision_model()
