import os
import platform
import torch


def get_device():
    """
    Auto-detect the best available accelerator.
    Priority: CUDA (NVIDIA GPU) > MPS (Apple Silicon) > CPU.
    Also applies backend-specific settings for maximum throughput.
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        gpu_name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_mem / (1024 ** 3)
        print(f"[Device] CUDA detected — {gpu_name} ({vram:.1f} GB)")

        # cuDNN auto-tuner: finds the fastest conv algorithm for the input size
        torch.backends.cudnn.benchmark = True

    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
        chip = platform.processor() or platform.machine()  # e.g. "arm"
        print(f"[Device] MPS detected — Apple Silicon ({chip})")

        # MPS memory: let PyTorch use up to 0 (unlimited) of unified memory.
        # Default caps at ~70 %; removing the cap avoids premature OOM errors
        # on machines with 18-48 GB unified RAM (M4 Pro, M4 Max, etc.).
        os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.0")

        # Some MPS ops still lack float16 kernels; this forces fallback to CPU
        # only for those individual ops rather than crashing.
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

    else:
        device = torch.device("cpu")
        print("[Device] No GPU found — using CPU")

    return device


def get_device_type(device: torch.device) -> str:
    """Return the string type: 'cuda', 'mps', or 'cpu'."""
    return device.type


def get_optimal_workers(device: torch.device | None = None):
    """
    DataLoader worker count tuned per backend.
    - CUDA  : 4  (GPU and CPU overlap via pinned memory + workers)
    - MPS   : 0  (macOS multiprocess + MPS can deadlock; main-process is safer)
    - CPU   : 0
    """
    if device is not None and device.type == "cuda":
        return 4
    if torch.cuda.is_available():
        return 4
    return 0


def get_pin_memory(device: torch.device) -> bool:
    """
    pin_memory speeds up host->device copies but only helps on CUDA.
    On MPS / CPU it is either ignored or causes warnings.
    """
    return device.type == "cuda"


def get_optimal_dtype(device: torch.device):
    """
    Default tensor dtype per backend.
    CUDA  : float32 (or float16 if you want AMP — handled separately)
    MPS   : float32 (float16 support is still partial on MPS)
    CPU   : float32
    """
    return torch.float32


def to_device(tensor, device):
    """Move a tensor (or a collection of tensors) to the given device."""
    if isinstance(tensor, (list, tuple)):
        return type(tensor)(to_device(t, device) for t in tensor)
    return tensor.to(device, non_blocking=(device.type == "cuda"))


def empty_cache(device: torch.device):
    """Release unused cached memory on the current accelerator."""
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "mps":
        torch.mps.empty_cache()
