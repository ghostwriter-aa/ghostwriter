import torch


def detect_device() -> str:
    """
    Detect the best available device for computation.

    Returns:
        Device string: "cuda" if CUDA GPU is available, "mps" if Apple Silicon GPU is available, "cpu" otherwise
    """
    if torch.cuda.is_available():
        device = "cuda"
        print(f"CUDA GPU detected: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        device = "mps"
        print("Apple Silicon GPU (MPS) detected")
    else:
        device = "cpu"
        print("No GPU detected, using CPU")

    print(f"Using device: {device}")
    return device
