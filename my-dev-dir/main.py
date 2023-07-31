import torch
import torch.distributed as dist
import os

print("{}".format(torch.cuda.is_available()))

dist.init_process_group("nccl","tcp://localhost:80",world_size=1,rank=0)
print(
dist.is_available()
)
