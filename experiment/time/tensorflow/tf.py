import os
import time
import tensorflow as tf
import sys

def init_gpu():
    """
    Initializes GPU and returns the list of available GPUs.
    If no GPU device is available, the program will exit.

    Returns:
        List[tf.config.experimental.PhysicalDevice]: A list of all available GPUs.
    """
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if not gpus: 
        print('No GPU available!', file=sys.stderr)
        sys.exit(1)
    return gpus

def init_context(gpus):
    """
    Initializes the context on the first GPU and returns the time it took.

    Args:
        gpus (List[tf.config.experimental.PhysicalDevice]): A list of all available GPUs.

    Returns:
        float: Time taken to initialize the context in seconds.
    """
    start_time = time.time()
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
    except RuntimeError as e:
        print(f"Error initializing GPU: {e}", file=sys.stderr)
        sys.exit(1)
    return time.time() - start_time

def allocate_tensor():
    """
    Allocates a random tensor and returns the time it took.

    Returns:
        float: Time taken to allocate the tensor in seconds.
    """
    start_time = time.time()
    a = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    return time.time() - start_time

def load_model():
    """
    Loads a model (e.g., ResNet) and returns the time it took.

    Returns:
        float: Time taken to load the model in seconds.
    """
    start_time = time.time()
    model = tf.keras.applications.ResNet50(weights='imagenet')
    return time.time() - start_time

def main():
    """
    Main function that orchestrates the initialization of GPU, context, tensor allocation, and model loading.
    It also prints the respective times each process took in milliseconds.
    """
    gpus = init_gpu()
    print('GPUs:', gpus)
    
    init_time = init_context(gpus)
    print('Context Initialization Time: ', init_time * 1000, "ms")
    
    tensor_alloc_time = allocate_tensor()
    print('Tensor Allocation Time: ', tensor_alloc_time * 1000, "ms")
    
    load_time = load_model()
    print('Model Load Time: ', load_time * 1000, "ms")

if __name__ == "__main__":
    main()
