from concurrent.futures import ThreadPoolExecutor
import time
from typing import Optional
import torch
import torch.nn as nn

import torch_xla.core.xla_model as xm
from library.device_utils import clean_memory_on_device

def print_block_size(block, name=""):
    total_size_in_bytes = 0
    for p in block.parameters():
        param_size_in_bytes = p.numel() * (2 if p.dtype == torch.bfloat16 else 4)  # Check dtype
        total_size_in_bytes += param_size_in_bytes
    size_in_mb = total_size_in_bytes / (1024**2)
    print(f"Block {name}: {size_in_mb:.2f} MB")

def synchronize_device(device: torch.device):
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "xla":
        xm.mark_step()  # Use xm.mark_step() for synchronization on TPUs
    elif device.type == "xpu":
        torch.xpu.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()

def swap_weight_devices_cuda(device: torch.device, layer_to_cpu: nn.Module, layer_to_cuda: nn.Module):
    assert layer_to_cpu.__class__ == layer_to_cuda.__class__

    weight_swap_jobs = []

    modules_to_cpu = {k: v for k, v in layer_to_cpu.named_modules()}
    for module_to_cuda_name, module_to_cuda in layer_to_cuda.named_modules():
        if hasattr(module_to_cuda, "weight") and module_to_cuda.weight is not None:
            module_to_cpu = modules_to_cpu.get(module_to_cuda_name, None)
            if module_to_cpu is not None and module_to_cpu.weight.shape == module_to_cuda.weight.shape:
                weight_swap_jobs.append((module_to_cpu, module_to_cuda, module_to_cpu.weight.data, module_to_cuda.weight.data))
            else:
                if module_to_cuda.weight.data.device.type != device.type:
                    module_to_cuda.weight.data = module_to_cuda.weight.data.to(device)

    torch.cuda.current_stream().synchronize()  # this prevents the illegal loss value

    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        # cuda to cpu
        for module_to_cpu, module_to_cuda, cuda_data_view, cpu_data_view in weight_swap_jobs:
            cuda_data_view.record_stream(stream)
            module_to_cpu.weight.data = cuda_data_view.data.to("cpu", non_blocking=True)

        stream.synchronize()

        # cpu to cuda
        for module_to_cpu, module_to_cuda, cuda_data_view, cpu_data_view in weight_swap_jobs:
            cuda_data_view.copy_(module_to_cuda.weight.data, non_blocking=True)
            module_to_cuda.weight.data = cuda_data_view

    stream.synchronize()
    torch.cuda.current_stream().synchronize()  # this prevents the illegal loss value

def swap_weight_devices_xla(device: torch.device, layer_to_cpu: nn.Module, layer_to_xla: nn.Module):
    """
    Swaps weights between CPU and XLA device (TPU core) for two layers.
    """
    assert layer_to_cpu.__class__ == layer_to_xla.__class__

    modules_to_cpu = {k: v for k, v in layer_to_cpu.named_modules()}

    for module_to_xla_name, module_to_xla in layer_to_xla.named_modules():
        if hasattr(module_to_xla, "weight") and module_to_xla.weight is not None:
            module_to_cpu = modules_to_cpu.get(module_to_xla_name, None)
            if (
                module_to_cpu is not None
                and hasattr(module_to_cpu, "weight") # Check if module_to_cpu also has a weight
                and module_to_cpu.weight is not None
                and module_to_cpu.weight.shape == module_to_xla.weight.shape
            ):
                print(f"Swapping weights for module: {module_to_xla_name}")

                module_to_cpu.weight.data, module_to_xla.weight.data = (
                    module_to_xla.weight.data,
                    module_to_cpu.weight.data,
                )
                module_to_cpu.weight.data = module_to_cpu.weight.data.to("cpu", non_blocking=False)
                module_to_xla.weight.data = module_to_xla.weight.data.to(device, non_blocking=False)

                print(f"Moved {module_to_xla_name}.weight to CPU")
                print(f"Moved {module_to_xla_name}.weight to {device}")
            else:
                if module_to_xla.weight.data.device.type != device.type:
                    module_to_xla.weight.data = module_to_xla.weight.data.to(device)
                    print(f"Moved {module_to_xla_name}.weight to {device}")

    xm.mark_step()
    print("XLA mark_step() called after weight swapping")

def weighs_to_device(layer: nn.Module, device: torch.device):
    layer.to(device, non_blocking=True)

class Offloader:
    """
    common offloading class
    """

    def __init__(self, num_blocks: int, blocks_to_swap: int, device: torch.device, debug: bool = False):
        self.num_blocks = num_blocks
        self.blocks_to_swap = blocks_to_swap
        self.device = device
        self.debug = debug

        self.thread_pool = ThreadPoolExecutor(max_workers=1)
        self.futures = {}
        self.xla_available = device.type == "xla"

    def swap_weight_devices(self, block_to_cpu: nn.Module, block_to_cuda: nn.Module):
        if self.xla_available:
            swap_weight_devices_xla(self.device, block_to_cpu, block_to_cuda)
        else:
            swap_weight_devices_cuda(self.device, block_to_cpu, block_to_cuda)

    def _submit_move_blocks(self, blocks, block_idx_to_cpu, block_idx_to_cuda):
        def move_blocks(bidx_to_cpu, block_to_cpu, bidx_to_cuda, block_to_cuda):
            if self.debug:
                start_time = time.perf_counter()
                print_block_size(block_to_cpu, name=f"CPU-bound (idx {bidx_to_cpu})")
                print_block_size(block_to_cuda, name=f"XLA-bound (idx {bidx_to_cuda})")
                print(f"Move block {bidx_to_cpu} to CPU and block {bidx_to_cuda} to XLA")

            self.swap_weight_devices(block_to_cpu, block_to_cuda)

            if self.debug:
                print(f"Moved blocks {bidx_to_cpu} and {bidx_to_cuda} in {time.perf_counter() - start_time:.2f}s")
            return bidx_to_cpu, bidx_to_cuda  # , event

        block_to_cpu = blocks[block_idx_to_cpu]
        block_to_cuda = blocks[block_idx_to_cuda]

        self.futures[block_idx_to_cuda] = self.thread_pool.submit(
            move_blocks, block_idx_to_cpu, block_to_cpu, block_idx_to_cuda, block_to_cuda
        )

    def _wait_blocks_move(self, block_idx):
        if block_idx not in self.futures:
            return

        if self.debug:
            print(f"Wait for block {block_idx}")
            start_time = time.perf_counter()

        future = self.futures.pop(block_idx)
        _, bidx_to_cuda = future.result()

        assert block_idx == bidx_to_cuda, f"Block index mismatch: {block_idx} != {bidx_to_cuda}"

        if self.debug:
            print(f"Waited for block {block_idx}: {time.perf_counter() - start_time:.2f}s")

class ModelOffloader(Offloader):
    """
    supports forward offloading
    """

    def __init__(
        self, blocks: list[nn.Module], num_blocks: int, blocks_to_swap: int, device: torch.device, debug: bool = False
    ):
        super().__init__(num_blocks, blocks_to_swap, device, debug)

        # register backward hooks
        self.remove_handles = []
        for i, block in enumerate(blocks):
            hook = self.create_backward_hook(blocks, i)
            if hook is not None:
                handle = block.register_full_backward_hook(hook)
                self.remove_handles.append(handle)

    def __del__(self):
        for handle in self.remove_handles:
            handle.remove()

    def create_backward_hook(self, blocks: list[nn.Module], block_index: int) -> Optional[callable]:
        # -1 for 0-based index
        num_blocks_propagated = self.num_blocks - block_index - 1
        swapping = num_blocks_propagated > 0 and num_blocks_propagated <= self.blocks_to_swap
        waiting = block_index > 0 and block_index <= self.blocks_to_swap

        if not swapping and not waiting:
            return None

        # create  hook
        block_idx_to_cpu = self.num_blocks - num_blocks_propagated
        block_idx_to_cuda = self.blocks_to_swap - num_blocks_propagated
        block_idx_to_wait = block_index - 1

        def backward_hook(module, grad_input, grad_output):
            if self.debug:
                print(f"Backward hook for block {block_index}")

            if swapping:
                self._submit_move_blocks(blocks, block_idx_to_cpu, block_idx_to_cuda)
            if waiting:
                self._wait_blocks_move(block_idx_to_wait)
            return None

        return backward_hook

    def prepare_block_devices_before_forward(self, blocks: list[nn.Module]):
        if self.blocks_to_swap is None or self.blocks_to_swap == 0:
            return

        if self.debug:
            print("Prepare block devices before forward")

        for i, b in enumerate(blocks[: self.num_blocks - self.blocks_to_swap]):
            print_block_size(b, name=f"Initial XLA-bound (idx {i})")
            # b.to(self.device)  # Remove this line
            weighs_to_device(b, self.device)

        for i, b in enumerate(blocks[self.num_blocks - self.blocks_to_swap :]):
            print_block_size(b, name=f"Initial CPU-bound (idx {i})")
            # b.to(self.device)  # Remove this line
            weighs_to_device(b, "cpu")

        synchronize_device(self.device)
        clean_memory_on_device(self.device)

    def wait_for_block(self, block_idx: int):
        if self.blocks_to_swap is None or self.blocks_to_swap == 0:
            return
        self._wait_blocks_move(block_idx)

    def submit_move_blocks(self, blocks: list[nn.Module], block_idx: int):
        if self.blocks_to_swap is None or self.blocks_to_swap == 0:
            return
        if block_idx >= self.blocks_to_swap:
            return

        xm.rendezvous("move_block_start")  # Add a synchronization barrier before moving blocks

        block_idx_to_cpu = block_idx
        block_idx_to_cuda = self.num_blocks - self.blocks_to_swap + block_idx
        self._submit_move_blocks(blocks, block_idx_to_cpu, block_idx_to_cuda)