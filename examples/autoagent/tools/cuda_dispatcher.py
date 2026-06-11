import os
import yaml
import torch
from torch.utils.dlpack import to_dlpack

class CUDADispatcher:
    """A generic bridge to dispatch CUDA kernels via tvm_ffi and DLPack."""
    def __init__(self, so_path, meta_path):
        import tvm_ffi
        self.mod = tvm_ffi.load_module(os.path.abspath(so_path))
        self.f = self.mod["dispatch_kernel"]
        with open(meta_path, "r") as f:
            self.active_kernel = yaml.safe_load(f)

    def __call__(self, *args):
        import tvm_ffi
        # Automatically wrap tensors in DLPack for zero-copy FFI
        wrapped_args = []
        for val in args:
            if isinstance(val, torch.Tensor):
                wrapped_args.append(tvm_ffi.from_dlpack(to_dlpack(val)))
            else:
                wrapped_args.append(val)
        
        self.f(*wrapped_args)
