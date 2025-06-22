import torch
import torch.distributed as dist


class DDP(torch.nn.Module):
    def __init__(self, module: torch.nn.Module):
        super().__init__()
        self.module = module
        self._handles = []
        self._grad_ready = []

        self._broadcast_parameters()
        self._register_hooks()

    def _broadcast_parameters(self):
        for param in self.module.parameters():
            dist.broadcast(param.data, src=0)

    def _register_hooks(self):
        def _hook(param):
            handle = dist.all_reduce(param.grad, op=dist.ReduceOp.SUM, async_op=True)
            self._handles.append((handle, param))
        
        for p in self.parameters():
            if p.requires_grad:
                p.register_post_accumulate_grad_hook(lambda _, p=p: _hook(p))

    def forward(self, *inputs, **kwargs):
        return self.module(*inputs, **kwargs)

    def finish_gradient_synchronization(self):
        world_size = dist.get_world_size()
        for handle, param in self._handles:
            handle.wait()
            param.grad.div_(world_size)
        self._handles.clear()

    def named_parameters(self, *args, **kwargs):
        return self.module.named_parameters(*args, **kwargs)

    def parameters(self, *args, **kwargs):
        return self.module.parameters(*args, **kwargs)
