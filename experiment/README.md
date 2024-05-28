## Cold Start Time 

### Description
This experiment measures the cold start time of a function. The cold start time is the time it takes for a function to be invoked for the first time. The experiment is repeated for a number of iterations. The result is the average of the cold start times of all iterations.

### Steps
1. Build the docker image and enter the function container
    ```bash
    cd $PATH_TO_STREAMBOX/experiment/time
    bash run.sh
    ```
2. In the container, tun the script
    ```bash
    bash run.sh
    bash stat.sh
    ```
    ```
    > bash run.sh
    Running iteration 1/10...
    Running iteration 2/10...
    ...
    Running iteration 10/10...
    > bash stat.sh
    Stats for context_init_times_torch:
    Median: 4481.4
    Average: 4472.76
    Min: 4107.215404510498
    Max: 4896.044015884399
    ...
    ```

### Analysis
According to our experimental results, the cold start latency of the GPU runtime exceeds 5 seconds, which aligns with our observations made in the paper. In this experiment, we analyzed the cold start time of a function, referring to the time it takes for the function to be invoked for the first time. The results show that a significant portion of the cold start time is consumed by the initialization of the GPU runtime. The average time taken was 4472.76 milliseconds, with the maximum time reaching up to 4896.04 milliseconds. These time durations are unacceptable.


## Memory Footprint

### Description
In this experiment, we evaluate the GPU memory footprint of serverless inference systems, particularly the function isolation approach.

### Steps

1. Run the script
    ```shell
    cd $PATH_TO_STREAMBOX/experiment/memory
    sudo bash run.sh
    ```
    Output:
    ```
    CUDA_VISIBLE_DEVICES=3
    ...
    Compute Mode < Exclusive Process >
    ...

    nvidia-cuda-mps-control daemon started
    Set MPS thread percentage to 10

    Successfully built a1b2c3d4e5f6
    Successfully tagged torch:streambox

    // run the docker containers
    a1b2c3d4e5f6

    CONTAINER ID   IMAGE           COMMAND                  CREATED          STATUS          PORTS     NAMES
    a1b2c3d4e5f6   torch:streambox   "python app.py"    3 seconds ago    Up 2 seconds              torch-mps-0
    a1b2c3d4e5f7   torch:streambox   "python app.py"    3 seconds ago    Up 2 seconds              torch-mps-1
    ...
    a1b2c3d4e5fn   torch:streambox   "python app.py"    3 seconds ago    Up 2 seconds              torch-mps-n

    // Logs from torch-mps-0:
    ...
    Driver Version: 465.19.01
    CUDA_VISIBLE_DEVICES: 3
    --Begin-- Current GPU memory info: Used: 500MB, Free: 15500MB, Total: 16000MB
    torch.cuda.current_device(): 0
    Running on Tesla V100-PCIE-16GB
    Number of GPUs: 1
    --After torch init-- Current GPU memory info: Used: 1900MB, Free: 15200MB, Total: 16000MB
    --After model init-- Current GPU memory info: Used: 2000MB, Free: 13500MB, Total: 16000MB
    Pytorch Context GPU Memory Usage: 1500MB

    // Logs from torch-mps-n:
    ...

    // Resetting MPS and Compute Mode...
    ```

### Analysis

Utilizing the DNN inference function (ResNet), we discover that the GPU runtime occupies over 95% of the overall memory footprint, which results in significant redundancy and low deployment density. Each function occpies 2GB of GPU memory, which is unacceptable for a serverless system. The memory footprint of the GPU runtime is a major bottleneck for the function isolation approach.

## Communication

### Function-IPC

#### Description
In this experiment, we evaluate the communication overhead of the function-IPC approach. You can measure the latency of the function-IPC approach under different message sizes.

#### Steps
1. Build the docker image and enter the function container
    ```bash
    cd $PATH_TO_STREAMBOX/experiment/communication/function-ipc
    bash run.sh
    ```
2. Output looks like:
    ```txt
    > bash run.sh
    ...
    torch.cuda.current_device(): 0
    Running on Tesla V100-SXM2-32GB
    Number of GPUs: 1
    Warming up GPU...
    GPU warmup done!
    Time taken to transfer data from device to host: xxx ms
    ...
    ```

### CUDA IPC

#### Description
In this experiment, we evaluate the communication overhead of the CUDA IPC approach. [CUDA IPC](https://forums.developer.nvidia.com/t/gpu-inter-process-communications-ipc-question/35936) is a mechanism that allows multiple processes to access a single GPU. [CUDA Runtime API](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__DEVICE.html#group__CUDART__DEVICE_1g8a37f7dfafaca652391d0758b3667539)

#### Steps
1. Build the docker image and enter the function container
    ```bash
    cd $PATH_TO_STREAMBOX/experiment/communication/cudaipc
    bash run.sh
    ```
2. Output file looks like:
    ```txt
    app1.txt
    ---
    get IPC handle1 time = 25.095000 us
    get IPC handle2 time = 15.128000 us
    waiting for app2
    data1 size = 268.435455 MB
    end-to-end time = 17632.712000 us
    ...
    app2.txt
    ---
    IPC transfer 2 times cost = 0.000000 us
    IPC open handle 1 time = 334.944000 us
    IPC open handle 2 time = 292.726013 us
    NO1 kernel + handle1 time = 40.992001 us
    NO2 kernel + handle1 time = 22.560001 us
    NO3 kernel + handle1 time = 23.360001 us
    NO1 kernel + handle2 time = 24.224001 us
    NO2 kernel + handle2 time = 21.472000 us
    NO3 kernel + handle2 time = 20.799999 us
    ...
    ```

