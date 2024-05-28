import os
import time
from pynvml import *

def get_gpu_mem_info(gpu_id=0):
    """
    Retrieves GPU memory information by GPU ID.

    Args:
        gpu_id (int, optional): GPU (device) ID. Defaults to 0.

    Returns:
        tuple: Total GPU memory, used GPU memory, free GPU memory (in MB).
    """
    try:
        if gpu_id < 0 or gpu_id >= nvmlDeviceGetCount():
            print(f"gpu_id {gpu_id} is not available")
            return 0, 0, 0
    except NVMLError as error:
        print(error)
    try:
        handler = nvmlDeviceGetHandleByIndex(gpu_id)
        meminfo = nvmlDeviceGetMemoryInfo(handler)
    except NVMLError as error:
        print(error)
    total = round(meminfo.total / 1024 / 1024, 2)
    used = round(meminfo.used / 1024 / 1024, 2)
    free = round(meminfo.free / 1024 / 1024, 2)

    return total, used, free

def main():
    """
    Main function that orchestrates the initialization of NVML, GPU memory info fetching, CUDA and PyTorch initialization, 
    and model loading. It also prints the respective GPU memory details at each stage.
    """
    try:
        nvmlInit()
        print("Driver Version:", nvmlSystemGetDriverVersion())
    except NVMLError as error:
        print(error)
    device_id = int(os.getenv("CUDA_VISIBLE_DEVICES"))
    print("CUDA_VISIBLE_DEVICES:", device_id)
    total_0, used_0, free_0 = get_gpu_mem_info(device_id)
    print(f"--Begin-- Current GPU memory info: Used: {used_0}MB, Free: {free_0}MB, Total: {total_0}MB")
    
    import torch
    from torchvision import models
    torch.cuda.init()
    # Check if CUDA is available
    if not torch.cuda.is_available():
        raise SystemError("CUDA is not available. This program needs to run on GPU.")
        
    # Get CUDA device
    device = torch.device("cuda")
    print("torch.cuda.current_device():", torch.cuda.current_device())
    # Print CUDA details
    print(f"Running on {torch.cuda.get_device_name(device)}")
    print(f"Number of GPUs: {torch.cuda.device_count()}")

    # after torch init
    total_t, used_t, free_t = get_gpu_mem_info(device_id)
    print(f"--After torch init-- Current GPU memory info: Used: {used_t}MB, Free: {free_t}MB, Total: {total_t}MB")
    
    # Initialize the model and move it to GPU
    model = models.resnet50(pretrained=True)
    model = model.to(device)

    # after model init
    total_f, used_f, free_f = get_gpu_mem_info(device_id)
    print(f"--After model init-- Current GPU memory info: Used: {used_f}MB, Free: {free_f}MB, Total: {total_f}MB")
    
    print(f"Pytorch Context GPU Memory Usage: {used_f - used_0}MB")
    
    time.sleep(10)

    nvmlShutdown()

if __name__ == "__main__":
    main()
