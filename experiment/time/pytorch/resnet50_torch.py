import torch
import time
import torchvision.models as models
import sys

def init_cuda():
    """
    Initializes CUDA and returns the device.
    If no CUDA device is available, the program will exit.

    Returns:
        device (torch.device): The CUDA device.
    """
    if torch.cuda.is_available(): 
        device = torch.device('cuda')
    else: 
        print('No GPU available!', file=sys.stderr)
        sys.exit(1)
    return device

def init_context(device):
    """
    Initializes the context and returns the time it took.

    Args:
        device (torch.device): The CUDA device.

    Returns:
        float: Time taken to initialize the context in seconds.
    """
    start_time = time.time()
    torch.cuda.init()
    torch.cuda.synchronize()
    return time.time() - start_time

def allocate_tensor(device):
    """
    Allocates a random tensor on the given device and returns the time it took.

    Args:
        device (torch.device): The CUDA device.

    Returns:
        float: Time taken to allocate the tensor in seconds.
    """
    start_time = time.time()
    tensor1 = torch.randn(1, device=device)
    torch.cuda.synchronize()
    return time.time() - start_time

def load_model(device):
    """
    Loads a model (e.g., ResNet) onto the given device and returns the time it took.

    Args:
        device (torch.device): The CUDA device.

    Returns:
        float: Time taken to load the model in seconds.
    """
    model = models.resnet50(pretrained=True) # 97.8M
    start_time = time.time()
    model.to(device)
    torch.cuda.synchronize()
    return time.time() - start_time

def main():
    device = init_cuda()
    print('Device:', device)
    
    init_time = init_context(device)
    print('Context Initialization Time: ', init_time * 1000, "ms")
    
    tensor1_time = allocate_tensor(device)
    print('Tensor1 Allocation Time: ', tensor1_time * 1000, "ms")
    
    load_time = load_model(device)
    print('Model Load Time: ', load_time * 1000, "ms")

if __name__ == "__main__":
    main()
