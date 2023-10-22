from typing import Tuple, Dict, Any

import torch
import torch.fx._pytree as fx_pytree
import torch.utils._pytree as pytree
from torch.export import export
from torch.fx import GraphModule, Interpreter
from torch.fx.node import Argument
from torchvision.models.resnet import resnet152

from compile_test import create_vision_dataloader, VisionModel


class PrintOperatorInterpreter(Interpreter):
    target_type_dict = dict()

    def call_function(self, target: 'Target', args: Tuple[Argument, ...], kwargs: Dict[str, Any]) -> Any:
        # target应是OpOverloadPacket或者OpOverload的实例
        target_type = str(type(target))
        print("target type:{}".format(target_type))
        if PrintOperatorInterpreter.target_type_dict.get(target_type) is None:
            PrintOperatorInterpreter.target_type_dict[target_type] = 0
        PrintOperatorInterpreter.target_type_dict[target_type] += 1
        for i, item in enumerate(dir(target)):
            if not item.startswith("__"):
                print("i:{},name:{},\nvalue:\n{}".format(i, item, getattr(target, item)))
        print("-------------------------------------------")
        return super().call_function(target, args, kwargs)

    # 从`ExportedProgram`的`__call__`方法处copy来
    @classmethod
    def print_exported_prog_operator(cls, exported_prog: torch.export.ExportedProgram, *args: Any,
                                     **kwargs: Any) -> Any:
        import torch._export.error as error
        from torch._export import combine_args_kwargs

        if exported_prog.call_spec.in_spec is not None:
            try:
                user_args = combine_args_kwargs(args, kwargs)
                args = fx_pytree.tree_flatten_spec(user_args,
                                                   exported_prog.call_spec.in_spec)  # type: ignore[assignment]
            except Exception:
                _, received_spec = pytree.tree_flatten(user_args)
                raise error.InternalError(
                    "Trying to flatten user inputs with exported input tree spec: \n"
                    f"{exported_prog.call_spec.in_spec}\n"
                    "but actually got inputs with tree spec of: \n"
                    f"{received_spec}"
                )

        ordered_params = tuple(
            exported_prog.state_dict[name] for name in exported_prog.graph_signature.parameters
        )
        ordered_buffers = tuple(
            exported_prog.state_dict[name] for name in exported_prog.graph_signature.buffers
        )
        exported_prog._check_input_constraints(*ordered_params, *ordered_buffers, *args)

        with torch.no_grad():
            # NOTE: calling convention is first params, then buffers, then args as user supplied them.
            # See: torch/_functorch/aot_autograd.py#L1034

            print_operator_interpreter = cls(exported_prog.graph_module)
            res = print_operator_interpreter.run(
                *ordered_params, *ordered_buffers, *args, enable_io_processing=False
            )
            print("target_type_dict:")
            print(PrintOperatorInterpreter.target_type_dict)

        if exported_prog.call_spec.out_spec is not None:
            mutation = exported_prog.graph_signature.buffers_to_mutate
            num_mutated = len(mutation)
            mutated_buffers = res[:num_mutated]

            # Exclude dependency token from final result.
            assertion_dep_token = exported_prog.graph_signature.assertion_dep_token
            if assertion_dep_token is not None:
                assertion_dep_token_index = list(assertion_dep_token.keys())[0]
                res = res[:assertion_dep_token_index]

            res = res[num_mutated:]
            try:
                res = pytree.tree_unflatten(res, exported_prog.call_spec.out_spec)
            except Exception:
                _, received_spec = pytree.tree_flatten(res)
                raise error.InternalError(
                    "Trying to flatten user outputs with exported output tree spec: \n"
                    f"{exported_prog.call_spec.out_spec}\n"
                    "but actually got outputs with tree spec of: \n"
                    f"{received_spec}"
                )
            finally:
                ix = 0
                for buffer in exported_prog.graph_signature.buffers_to_mutate.values():
                    exported_prog.state_dict[buffer] = mutated_buffers[ix]
                    ix += 1
        return res


def test_torch_export(m: torch.nn.Module, *args):
    args = tuple(args)
    exported_program: torch.export.ExportedProgram = export(
        m, args
    )

    print("compare fw output value:")
    gm: GraphModule = exported_program.graph_module
    gm.to("cuda:0")
    output = exported_program(*args)
    print(output)
    with torch.no_grad():
        output = m(*args)
        print(output)

    print("print operator attributes:")
    PrintOperatorInterpreter.print_exported_prog_operator(exported_program, *args)


if __name__ == '__main__':
    device = torch.device("cuda:0")
    model = resnet152().to(device)
    optim = torch.optim.Adam(params=model.parameters())
    loss_fn = torch.nn.CrossEntropyLoss().to(device)
    dataloader = create_vision_dataloader(64, 1000, device)
    vision_model = VisionModel(model, optim, loss_fn)

    test_torch_export(vision_model, *next(dataloader))
