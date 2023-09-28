import torch
from torchvision.models.resnet import resnet152
from transformers import BertConfig, BertForSequenceClassification, BertTokenizer
from typing import Iterator, Generator, Tuple, Callable


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


def benchmark(epoch: int, dataloader: Iterator[Tuple[torch.Tensor, torch.Tensor]],
              original_model: Callable[[torch.Tensor, torch.Tensor], None],
              compile_model: Callable[[torch.Tensor, torch.Tensor], None],
              ):
    # warm up
    for _ in range(5):
        data, label = next(dataloader)
        compile_model(data, label)
        original_model(data, label)
    torch.cuda.synchronize()

    timer = CudaEventTimer()
    total_speed_up_time = 0

    for i in range(epoch):
        data, label = next(dataloader)

        with timer:
            compile_model(data, label)
        compile_model_spend_time = timer.get_spend_time()
        print("epoch:{}, compile model spend time:{}ms".format(i + 1, compile_model_spend_time))

        with timer:
            original_model(data, label)
        original_model_spend_time = timer.get_spend_time()
        print("epoch:{}, original model spend time:{}ms".format(i + 1, original_model_spend_time))

        speed_up = original_model_spend_time - compile_model_spend_time
        print("epoch:{}, speed up:{}ms".format(i + 1, speed_up))
        total_speed_up_time += speed_up

    print("avg speed up time:{}".format(total_speed_up_time / epoch))


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


def compile_vision_model(epoch: int = 10):
    device = torch.device("cuda:0")
    model = resnet152().to(device)
    optim = torch.optim.Adam(params=model.parameters())
    loss_fn = torch.nn.CrossEntropyLoss().to(device)
    dataloader = create_vision_dataloader(64, 1000, device)

    vision_model = VisionModel(model, optim, loss_fn)
    compile_model = torch.compile(vision_model, mode="max-autotune")

    benchmark(epoch, dataloader, lambda data, label: vision_model.forward(data, label), compile_model)


class BertLanguageModel(torch.nn.Module):
    def __init__(self, model: BertForSequenceClassification, optim: torch.optim.Optimizer, loss_fn: torch.nn.Module,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = model
        self.optim = optim
        self.loss_fn = loss_fn

    def forward(self, data: torch.Tensor, label: torch.Tensor):
        self.optim.zero_grad()
        output = self.model.forward(data)
        output = torch.nn.functional.softmax(output.logits, dim=-1)
        loss: torch.Tensor = self.loss_fn.forward(output, label)
        loss.backward()
        self.optim.step()


def create_language_dataloader(batch_size: int, class_num: int, tokenizer: BertTokenizer, device: torch.device) -> \
        Generator[
            Tuple[torch.Tensor, torch.Tensor], None, None]:
    text = "Replace me by any text you'd like."

    batch_input = []
    for _ in range(batch_size):
        tokens = tokenizer.tokenize(text)
        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        batch_input.append(input_ids)
    batch_input = torch.tensor(batch_input).to(device)

    while 1:
        yield (
            batch_input, torch.randint(class_num, (batch_size,)).to(device)
        )


def compile_language_model(epoch: int = 10):
    device = torch.device("cuda:0")
    config = BertConfig.from_json_file("./bert_base_uncased_model_config.json")

    tokenizer: BertTokenizer = BertTokenizer("./vocab.txt", do_lower_case=True)
    model: BertForSequenceClassification = BertForSequenceClassification(config)
    model = model.to(device)

    language_model = BertLanguageModel(model, torch.optim.Adam(model.parameters()),
                                       torch.nn.CrossEntropyLoss().to(device))
    compile_model = torch.compile(language_model, mode="max-autotune")
    dataloader = create_language_dataloader(5, 2, tokenizer, device)

    benchmark(epoch, dataloader, lambda data, label: language_model.forward(data, label), compile_model)


if __name__ == '__main__':
    torch.set_float32_matmul_precision("high")
    compile_vision_model(10)
    compile_language_model(10)
