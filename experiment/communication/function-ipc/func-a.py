# func-a.py
import socket
import struct
import torch
import numpy as np

class TensorSender:
    """
    A class for sending PyTorch tensors through sockets. 

    Attributes:
        host (str): The host name or IP address of the socket.
        port (int): The port number to use, between 1 and 65535.
        tensor (torch.Tensor): The PyTorch tensor to be sent.
    """

    def __init__(self, host, port, tensor):
        """
        Constructs all the necessary attributes for the TensorSender object.

        Args:
            host (str): The host name or IP address of the socket.
            port (int): The port number to use, between 1 and 65535.
            tensor (torch.Tensor): The PyTorch tensor to be sent.
        """
        self.host = host
        self.port = port
        self.tensor = tensor

    def _create_socket(self):
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.connect((self.host, self.port))
        return sock

    def _send_array(self, sock, array):
        sock.sendall(struct.pack('<I', array.nbytes))
        sock.sendall(struct.pack('<I', len(array.shape)))
        for dimension in array.shape:
            sock.sendall(struct.pack('<I', dimension))
        sock.sendall(array.tobytes())

    def send_tensor(self):
        sock = self._create_socket()
        array = self.tensor.cpu().numpy()
        self._send_array(sock, array)
        sock.close()


if __name__ == '__main__':
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

    host = 'localhost'
    port = 12345
    tensor = torch.randn(1000, 1000).cuda()

    sender = TensorSender(host, port, tensor)
    sender.send_tensor()
