from typing import Iterator, Generator, Tuple, Callable, List, Optional, Dict, Any

import torch
import playground
from functorch.compile import make_boxed_func
from torch._dynamo import config
from torch._dynamo.backends.common import aot_autograd
from torch._ops import OpOverload
from torch.fx import GraphModule
from torch.fx import Interpreter
from torch.fx.node import Argument
from torchvision.models.resnet import resnet152
from transformers import BertConfig, BertForSequenceClassification, BertTokenizer


def default_compile_fn(m: torch.nn.Module) -> Callable:
    return torch.compile(m, mode="max-autotune")


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


def benchmark(
        epoch: int,
        dataloader: Iterator[Tuple[torch.Tensor, torch.Tensor]],
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
        print(
            "epoch:{}, compile model spend time:{}ms".format(
                i + 1, compile_model_spend_time
            )
        )

        with timer:
            original_model(data, label)
        original_model_spend_time = timer.get_spend_time()
        print(
            "epoch:{}, original model spend time:{}ms".format(
                i + 1, original_model_spend_time
            )
        )

        speed_up = original_model_spend_time - compile_model_spend_time
        print("epoch:{}, speed up:{}ms".format(i + 1, speed_up))
        total_speed_up_time += speed_up

    print("avg speed up time:{}".format(total_speed_up_time / epoch))


class VisionModel(torch.nn.Module):
    def __init__(
            self,
            model: torch.nn.Module,
            optim: torch.optim.Optimizer,
            loss_fn: torch.nn.Module,
            *args,
            **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.model = model
        # self.optim = optim
        self.loss_fn = loss_fn

    def forward(self, data, label):
        # self.optim.zero_grad()
        output: torch.Tensor = self.model(data)
        loss: torch.Tensor = self.loss_fn(output, label)
        # loss.backward()
        # self.optim.step()
        return loss


def create_vision_dataloader(
        batch_size: int, class_num: int, device: torch.device
) -> Generator[Tuple[torch.Tensor, torch.Tensor], None, None]:
    while 1:
        yield (
            torch.randn(batch_size, 3, 128, 128).to(torch.float32).to(device),
            torch.randint(class_num, (batch_size,)).to(device),
        )


def compile_vision_model(
        epoch: int = 10,
        compile_fn: Callable[
            [torch.nn.Module, Optional[List]], Callable
        ] = default_compile_fn,
):
    device = torch.device("cuda:0")
    model = resnet152().to(device)
    optim = torch.optim.Adam(params=model.parameters())
    loss_fn = torch.nn.CrossEntropyLoss().to(device)
    dataloader = create_vision_dataloader(64, 1000, device)

    vision_model = VisionModel(model, optim, loss_fn)

    compile_model = compile_fn(vision_model, list(next(dataloader)))

    benchmark(
        epoch,
        dataloader,
        lambda data, label: vision_model.forward(data, label),
        compile_model,
    )


class BertLanguageModel(torch.nn.Module):
    def __init__(
            self,
            model: BertForSequenceClassification,
            optim: torch.optim.Optimizer,
            loss_fn: torch.nn.Module,
            *args,
            **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.model = model
        self.optim = optim
        self.loss_fn = loss_fn

    def forward(self, data, label):
        # self.optim.zero_grad()
        output = self.model.forward(data)
        output = torch.nn.functional.softmax(output.logits, dim=-1)
        loss: torch.Tensor = self.loss_fn.forward(output, label)
        loss.backward()
        # self.optim.step()


def create_language_dataloader(
        batch_size: int, class_num: int, tokenizer: BertTokenizer, device: torch.device
) -> Generator[Tuple[torch.Tensor, torch.Tensor], None, None]:
    text = "Replace me by any text you'd like."

    batch_input = []
    for _ in range(batch_size):
        tokens = tokenizer.tokenize(text)
        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        batch_input.append(input_ids)
    batch_input = torch.tensor(batch_input).to(device)

    while 1:
        yield (batch_input, torch.randint(class_num, (batch_size,)).to(device))


def compile_language_model(
        epoch: int = 10,
        compile_fn: Callable[
            [torch.nn.Module, Optional[List]], Callable
        ] = default_compile_fn,
):
    device = torch.device("cuda:0")
    config = BertConfig.from_json_file("./bert_base_uncased_model_config.json")

    tokenizer: BertTokenizer = BertTokenizer("./vocab.txt", do_lower_case=True)
    dataloader = create_language_dataloader(5, 2, tokenizer, device)

    model: BertForSequenceClassification = BertForSequenceClassification(config)
    model = model.to(device)

    language_model = BertLanguageModel(
        model,
        torch.optim.Adam(model.parameters()),
        torch.nn.CrossEntropyLoss().to(device),
    )

    compile_model = compile_fn(language_model, list(next(dataloader)))

    benchmark(
        epoch,
        dataloader,
        lambda data, label: language_model.forward(data, label),
        compile_model,
    )


class GetOperatorInfoInterpreter(Interpreter):
    def call_function(
            self, target: "Target", args: Tuple[Argument, ...], kwargs: Dict[str, Any]
    ) -> Any:
        if isinstance(target, OpOverload):
            # 将target替换为hacking的ops
            namespace, name = target._schema.name.split("::")
            target = (
                playground.ops.__getattr__(namespace)
                .__getattr__(name)
                .__getattr__(target._overloadname)
            )
        return super().call_function(target, args, kwargs)

    @classmethod
    def collect_operator_info(cls, gm: GraphModule, *args, **kwargs):
        cls(gm).run(*args, **kwargs)


def graph_printer(gm: GraphModule, flag: str = ""):
    if not hasattr(graph_printer, "graph_print_count"):
        graph_printer.__setattr__("graph_print_count", 0)
    if not hasattr(graph_printer, "target_type_dict"):
        graph_printer.__setattr__("target_type_dict", dict())
    graph_printer.graph_print_count += 1
    print(
        "{}graph print count:{}".format(
            "flag:{} ".format(flag) if flag != "" else "",
            graph_printer.graph_print_count,
        )
    )
    for node in gm.graph.nodes:
        # print("opcode:{},name:{},target:{},args:{}".format(node.op, node.name, node.target, node.args))
        if node.op == "call_function":
            target = node.target
            target_type = str(type(target))
            print("target type:{}".format(target_type))
            for i, item in enumerate(dir(target)):
                if not item.startswith("__"):
                    print(
                        "i:{},name:{},\nvalue:\n{}".format(
                            i, item, getattr(target, item)
                        )
                    )
            print("-------------------------------------------")
            if graph_printer.target_type_dict.get(target_type) is None:
                graph_printer.target_type_dict[target_type] = 0
            graph_printer.target_type_dict[target_type] += 1


def print_compile_fn(m: torch.nn.Module, args: Optional[List] = None) -> Callable:
    if args is not None:
        # explanation = dynamo.explain(m)(*args)
        # print(explanation)
        pass

    def create_print_fn(
            flag: str = "",
    ) -> Callable[[GraphModule, List[torch.Tensor]], Callable]:
        def print_fn(gm: GraphModule, inputs: List[torch.Tensor]):
            # graph_printer(gm, flag)
            GetOperatorInfoInterpreter.collect_operator_info(gm, *inputs)
            return make_boxed_func(gm.forward)

        return print_fn

    backend = aot_autograd(
        fw_compiler=create_print_fn("forward"), bw_compiler=create_print_fn("backward")
    )
    return torch.compile(m, backend=backend)


if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")
    config.verbose = True

    compile_vision_model(1, print_compile_fn)

    from playground._C import get_operators_info


    class Integer:
        def __init__(self, init: int):
            self.inner = init

        def add(self, num: int):
            self.inner += num

        def __str__(self):
            return str(self.inner)


    operator_overloaded_info: Dict[str, Tuple[Integer, List[List]]] = dict()
    for (op_type, (symbol, overloaded_args)) in get_operators_info():
        if operator_overloaded_info.get(symbol) is None:
            operator_overloaded_info[symbol] = (Integer(0), [])
        if op_type == "Overloaded":
            operator_overloaded_info[symbol][1].append(overloaded_args)
        else:
            operator_overloaded_info[symbol][0].add(1)

    overloaded_op_count = 0
    non_overloaded_op_count = 0
    for (symbol, (non_overloaded_count, overloaded_args)) in operator_overloaded_info.items():
        print("symbol: {}, non overloaded count: {}".format(symbol, non_overloaded_count))
        print("overloaded args: {}", overloaded_args)
        if len(overloaded_args) == 0:
            non_overloaded_op_count += 1
        else:
            overloaded_op_count += 1

    print("overloaded op count: {}".format(overloaded_op_count))
    print("non overloaded op count: {}".format(non_overloaded_op_count))
    # print("target_type_dict:")
    # print(graph_printer.target_type_dict)
    # compile_language_model(1, print_compile_fn)