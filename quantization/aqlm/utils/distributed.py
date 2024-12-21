import torch.distributed

def get_rank():
    if torch.cuda.is_available() and torch.distributed.is_available():
        if not torch.distributed.is_initialized():
            try:
                torch.distributed.init_process_group(backend='nccl')
            except ValueError:
                return 0
        rank = torch.distributed.get_rank()
    else:
        rank = 0
    return rank