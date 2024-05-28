# func-b.py
import socket
import struct
import torch
import numpy as np

class TensorReceiver:
    """
    A class for receiving PyTorch tensors through sockets. 

    Attributes:
        host (str): The host name or IP address of the socket.
        port (int): The port number to use, between 1 and 65535.
    """

    def __init__(self, host, port):
        """
        Constructs all the necessary attributes for the TensorReceiver object.

        Args:
            host (str): The host name or IP address of the socket.
            port (int): The port number to use, between 1 and 65535.
        """
        self.host = host
        self.port = port

    def _create_socket(self):
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.bind((self.host, self.port))
        sock.listen()
        return sock

    def _recv_array(self, sock):
        nbytes, = struct.unpack('<I', sock.recv(4))
        ndim, = struct.unpack('<I', sock.recv(4))
        shape = []
        for _ in range(ndim):
            dimension, = struct.unpack('<I', sock.recv(4))
            shape.append(dimension)
        shape = tuple(shape)
        array = np.frombuffer(sock.recv(nbytes), dtype=np.float32).reshape(shape)
        return array

    def recv_tensor(self):
        sock = self._create_socket()
        conn, _ = sock.accept()
        array = self._recv_array(conn)
        conn.close()
        sock.close()
        tensor = torch.from_numpy(array).cuda()
        return tensor

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

    receiver = TensorReceiver(host, port)
    tensor = receiver.recv_tensor()
