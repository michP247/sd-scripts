from concurrent.futures import ThreadPoolExecutor
import time
from typing import Optional
import torch
import torch_xla
import torch.nn as nn
from library.device_utils import clean_memory_on_device
import torch_xla.core.xla_model as xm
from typing import Optional, Tuple

import subprocess

def get_tpu_memory_info():
    """Retrieves and parses the output of the 'tpu-info' command."""
    try:
        t = torch.randn((300, 300), device=torch_xla.device())
        result = subprocess.run(['tpu-info', '-d', 'all'], capture_output=True, text=True, check=True)
        return result.stdout
    except subprocess.CalledProcessError as e:
        print(f"Error executing tpu-info: {e}")
        return "Could not retrieve TPU info."

def print_tpu_memory_usage():
    """Prints the TPU memory usage information."""
    print(get_tpu_memory_info())

def get_memory_stats(device: torch.device) -> str:
    """Gets a string summarizing memory statistics for the given device."""
    if device.type == 'xla':
        return xm.get_memory_info(device)
    else:
        return "Memory stats not available for non-XLA devices."

def print_memory_usage(device: torch.device):
    """Prints the memory usage for the given device."""
    print(get_memory_stats(device))
    
def get_tensor_size_bytes(tensor: torch.Tensor) -> int:
    """Calculates the size of a tensor in bytes."""
    return tensor.element_size() * tensor.nelement()

def get_module_size_bytes(module: nn.Module) -> int:
    """Estimates the size of a module in bytes."""
    size = 0
    for param in module.parameters():
        size += get_tensor_size_bytes(param)
    for buffer in module.buffers():
        size += get_tensor_size_bytes(buffer)
    return size

def synchronize_device(device: Optional[torch.device] = None):
    if device is not None and device.type == 'xla':
        xm.mark_step()
    elif device is not None and device.type == 'cuda':
        torch.cuda.synchronize()

def clean_memory_on_device(device: Optional[torch.device] = None):
    if device is not None and device.type == 'xla':
        xm.mark_step()
    elif device is not None and device.type == 'cuda':
        torch.cuda.empty_cache()

def swap_weight_devices_no_cuda(device: torch.device, layer_to_cpu: nn.Module, layer_to_device: nn.Module):
    """
    Swaps the weights of two layers between a non-CUDA device and CPU.

    This function assumes that `layer_to_device` is already on the specified device
    and `layer_to_cpu` is on the CPU.
    """
    assert layer_to_cpu.__class__ == layer_to_device.__class__

    # Transfer weights from device to CPU
    for module_to_cpu, module_to_device in zip(layer_to_cpu.modules(), layer_to_device.modules()):
        if hasattr(module_to_cpu, "weight") and module_to_cpu.weight is not None:
            # Ensure the module on the device has its weights transferred to CPU
            module_to_cpu.weight = nn.Parameter(module_to_device.weight.to("cpu", non_blocking=True))

    synchronize_device(device)

    # Transfer weights from CPU to device (synchronous)
    for module_to_cpu, module_to_device in zip(layer_to_cpu.modules(), layer_to_device.modules()):
        if hasattr(module_to_device, "weight") and module_to_device.weight is not None:
            module_to_device.weight = nn.Parameter(module_to_cpu.weight.to(device, non_blocking=False))  # Removed non_blocking

    synchronize_device(device)

def swap_weight_devices_cuda(device: torch.device, layer_to_cpu: nn.Module, layer_to_cuda: nn.Module):
    """
    Swaps the weights of two layers between GPU and CPU using a temporary tensor.

    This function assumes that `layer_to_device` is already on the GPU
    and `layer_to_cpu` is on the CPU.
    """
    assert layer_to_cpu.__class__ == layer_to_cuda.__class__

    for module_to_cpu, module_to_cuda in zip(layer_to_cpu.modules(), layer_to_cuda.modules()):
        if hasattr(module_to_cpu, "weight") and module_to_cpu.weight is not None:
            # Create a temporary tensor on the CPU to hold the weights
            temp_tensor = module_to_cpu.weight.to(device, non_blocking=True)

            # Move the weights from GPU to CPU
            module_to_cpu.weight = nn.Parameter(module_to_cuda.weight.to("cpu", non_blocking=True))
            synchronize_device(device)  # Ensure CPU move is complete

            # Move the weights from the temporary CPU tensor to GPU
            module_to_cuda.weight = nn.Parameter(temp_tensor)
            synchronize_device(device)  # Ensure GPU move is complete

def weighs_to_device(layer: nn.Module, device: torch.device):
    for module in layer.modules():
        if hasattr(module, "weight") and module.weight is not None:
            #print(f"  Old device: {module.weight.data.device}")
            #print(f"  Old dtype: {module.weight.data.dtype}")
            #print(f"Moving module '{module}' to {device}")
            module.weight = torch.nn.Parameter(module.weight.to(device, non_blocking=True)) # Move entire parameter to CPU
            #print(f"  New device: {module.weight.data.device}")
            #print(f"  New dtype: {module.weight.data.dtype}")

class Offloader:
    """
    common offloading class
    """

    def __init__(self, device: torch.device, debug: bool = False) -> None:
        self.device = device
        self.debug = debug
        self.cuda_available = torch.cuda.is_available()

    def _submit_move_blocks(self, bidx_to_cpu, block_to_cpu, bidx_to_cuda, block_to_cuda) -> Tuple[int, int]:
        """
        Submits the move of two blocks to different devices.
        Returns the block indices of the blocks that were moved.
        """

        def move_blocks(bidx_to_cpu, block_to_cpu, bidx_to_cuda, block_to_cuda):
            if self.debug:
                if block_to_cpu is not None:
                    size_to_cpu = get_module_size_bytes(block_to_cpu)
                    print(f"Block {bidx_to_cpu} to CPU: Size = {size_to_cpu / (1024**2):.2f} MB")
                if block_to_cuda is not None:
                    size_to_cuda = get_module_size_bytes(block_to_cuda)
                    print(f"Block {bidx_to_cuda} to {'CUDA' if self.cuda_available else 'device'}: Size = {size_to_cuda / (1024**2):.2f} MB")
                print_memory_usage(self.device)

            start_time = time.perf_counter()
            print(f"Move block {bidx_to_cpu} to CPU and block {bidx_to_cuda} to {'CUDA' if self.cuda_available else 'device'}")

            # Add tpu-info output before and after the swap
            if self.device.type == 'xla':
                print("TPU Memory Info (Before Swap):")
                print_tpu_memory_usage()

            self.swap_weight_devices(block_to_cpu, block_to_cuda)

            if self.device.type == 'xla':
                print("TPU Memory Info (After Swap):")
                print_tpu_memory_usage()

            if self.debug:
                print(f"Moved blocks {bidx_to_cpu} and {bidx_to_cuda} in {time.perf_counter()-start_time:.2f}s")
                print_memory_usage(self.device)

            return bidx_to_cpu, bidx_to_cuda

        import concurrent.futures

        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(move_blocks, bidx_to_cpu, block_to_cpu, bidx_to_cuda, block_to_cuda)
        return future

    def _wait_blocks_move(self, block_idx):
        if block_idx not in self.futures:
            return
        future = self.futures.pop(block_idx)

        if self.debug:
            start_time = time.perf_counter()
            print(f"Wait for block {block_idx}")

        _, bidx_to_cuda = future.result()

        xm.mark_step()

        assert block_idx == bidx_to_cuda, f"Block index mismatch: {block_idx} != {bidx_to_cuda}"

        if self.debug:
            print(f"Waited for block {bidx_to_cuda}: {time.perf_counter()-start_time:.2f}s")
            print_memory_usage(self.device)

    def swap_weight_devices(self, block_to_cpu, block_to_cuda):
        if self.cuda_available:
            swap_weight_devices_cuda(self.device, block_to_cpu, block_to_cuda)
        else:
            swap_weight_devices_no_cuda(self.device, block_to_cpu, block_to_cuda)

class ModelOffloader(Offloader):
    """
    supports forward offloading
    """

    def __init__(self, blocks: list[nn.Module], num_blocks: int, blocks_to_swap: int, device: torch.device, debug: bool = False) -> None:
        super().__init__(device, debug)
        self.blocks = blocks
        self.num_blocks = num_blocks
        self.blocks_to_swap = blocks_to_swap
        self.futures = {}  # For storing futures of block moves

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
        block_idx_to_cuda = self.blocks_to_swap - num_blocks_propagated - 1
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

    # add this new method
    def prepare_for_block(self, block_idx: int):
        if self.blocks_to_swap is None or self.blocks_to_swap == 0:
            return
        if block_idx < self.num_blocks - self.blocks_to_swap:
            return
        self._wait_blocks_move(block_idx)

    def move_to_device_for_block(self, block: nn.Module, device: torch.device):
        # Check if the block has any parameters before attempting to access .weight
        if any(p.requires_grad for p in block.parameters()):
            # Find the first parameter and check its device
            first_param = next((p for p in block.parameters()), None)
            if first_param is not None and first_param.device != device:
                print(f"Moving block of type {block.__class__.__name__} to {device}")
                block.to(device)
        else:
            print(f"Block of type {block.__class__.__name__} has no parameters to move.")

    def prepare_block_devices_before_forward(self, blocks: list[nn.Module]):
        if self.blocks_to_swap is None or self.blocks_to_swap == 0:
            return

        if self.debug:
            print("Prepare block devices before forward")
            for i, b in enumerate(blocks):
                if hasattr(b, "img_mod") and hasattr(b.img_mod, "lin") and hasattr(b.img_mod.lin, "weight"):
                    print(f"  Parameter 'double_blocks.{i}.img_mod.lin.weight' device: {b.img_mod.lin.weight.device}")

        for b in blocks[0 : self.num_blocks - self.blocks_to_swap]:
            weighs_to_device(b, self.device)  # make sure weights are on device

        for b in blocks[self.num_blocks - self.blocks_to_swap :]:
            weighs_to_device(b, torch.device("cpu"))  # make sure weights are on cpu, using torch.device("cpu")

        synchronize_device(self.device)
        clean_memory_on_device(self.device)

    def prepare_block_devices_before_backward(self, blocks: list[nn.Module]):
        if self.blocks_to_swap is None or self.blocks_to_swap == 0:
            return

        if self.debug:
            print("Prepare block devices before backward")

        # move blocks to cpu except the first blocks_to_swap blocks
        for b in blocks[self.blocks_to_swap :]:
            weighs_to_device(b, torch.device("cpu"))  # make sure weights are on cpu

        synchronize_device(self.device)
        clean_memory_on_device(self.device)

    def offload_block_weights(self, block_idx: int):
        if self.debug:
            print(f"Offloading weights for block {block_idx}")
            if hasattr(self.blocks[block_idx], "img_mod") and hasattr(self.blocks[block_idx].img_mod, "lin") and hasattr(self.blocks[block_idx].img_mod.lin, "weight"):
                print(f"  Parameter 'double_blocks.{block_idx}.img_mod.lin.weight' device before offloading: {self.blocks[block_idx].img_mod.lin.weight.device}")

        if block_idx >= self.num_blocks - self.blocks_to_swap:
            # Offload the current block to CPU
            weighs_to_device(self.blocks[block_idx], torch.device("cpu"))
            synchronize_device(self.device)

            if self.debug:
                print(f"  Parameter 'double_blocks.{block_idx}.img_mod.lin.weight' device after offloading to CPU: {self.blocks[block_idx].img_mod.lin.weight.device}")

        # Schedule the next block to be moved to the device if it exists
        next_block_idx = block_idx + self.blocks_to_swap
        if next_block_idx < self.num_blocks:
            self.futures[next_block_idx] = self._submit_move_blocks(block_idx, self.blocks[block_idx], next_block_idx, self.blocks[next_block_idx])

        if self.debug:
            if hasattr(self.blocks[block_idx], "img_mod") and hasattr(self.blocks[block_idx].img_mod, "lin") and hasattr(self.blocks[block_idx].img_mod.lin, "weight"):
                print(f"  Parameter 'double_blocks.{block_idx}.img_mod.lin.weight' device after move: {self.blocks[block_idx].img_mod.lin.weight.device}")
            if next_block_idx < self.num_blocks and hasattr(self.blocks[next_block_idx], "img_mod") and hasattr(self.blocks[next_block_idx].img_mod, "lin") and hasattr(self.blocks[next_block_idx].img_mod.lin, "weight"):
                print(f"  Parameter 'double_blocks.{next_block_idx}.img_mod.lin.weight' device after move: {self.blocks[next_block_idx].img_mod.lin.weight.device}")

    def offload_block_backward_hook(self, block_idx: int):
        def hook(*args, **kwargs):
            if self.debug:
                print(f"Offloading weights for block {block_idx} in backward pass")
            self.offload_block_weights(block_idx)

        return hook

    def wait_for_block(self, block_idx: int):
        if self.blocks_to_swap is None or self.blocks_to_swap == 0:
            return
        self._wait_blocks_move(block_idx)

    def submit_move_blocks(self, blocks: list[nn.Module], block_idx: int):
        if self.blocks_to_swap is None or self.blocks_to_swap == 0:
            return
        if block_idx >= self.blocks_to_swap:
            return
        block_idx_to_cpu = block_idx
        block_idx_to_cuda = self.num_blocks - self.blocks_to_swap + block_idx
        block_to_cuda = blocks[block_idx_to_cuda]  # Get the block to be moved to CUDA
        self._submit_move_blocks(block_idx_to_cpu, blocks[block_idx_to_cpu], block_idx_to_cuda, block_to_cuda)