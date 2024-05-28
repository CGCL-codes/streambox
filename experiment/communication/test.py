import socket
import struct
import time
import torch
import numpy as np
from multiprocessing import Process

class TensorTransfer(Process):
    """
    A class for sending and receiving PyTorch tensors through sockets. 
    It's derived from multiprocessing.Process class, therefore it can run 
    in a separate process.

    Attributes:
        host (str): The host name or IP address of the socket.
        port (int): The port number to use, between 1 and 65535.
        tensor (torch.Tensor): The PyTorch tensor to be sent or received.
        is_sender (bool): Specifies if this object is a sender.
    """

    def __init__(self, host, port, tensor=None):
        """
        Constructs all the necessary attributes for the TensorTransfer object.

        Args:
            host (str): The host name or IP address of the socket.
            port (int): The port number to use, between 1 and 65535.
            tensor (torch.Tensor, optional): The PyTorch tensor to be sent, if this object is a sender.
        """
        super().__init__()
        self.host = host
        self.port = port
        self.tensor = tensor
        self.is_sender = tensor is not None

    def run(self):
        if self.is_sender:
            self._send_tensor()
        else:
            self._recv_tensor()

    def _create_socket(self, server=False):
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        if server:
            sock.bind((self.host, self.port))
            sock.listen()
        else:
            sock.connect((self.host, self.port))
        return sock

    def _send_array(self, sock, array):
        sock.sendall(struct.pack('<I', array.nbytes))
        sock.sendall(struct.pack('<I', len(array.shape)))
        for dimension in array.shape:
            sock.sendall(struct.pack('<I', dimension))
        sock.sendall(array.tobytes())

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

    def _send_tensor(self):
        sock = self._create_socket()
        array = self.tensor.cpu().numpy()
        self._send_array(sock, array)
        sock.close()

    def _recv_tensor(self):
        sock = self._create_socket(server=True)
        conn, _ = sock.accept()
        array = self._recv_array(conn)
        conn.close()
        sock.close()
        self.tensor = torch.from_numpy(array).cuda()

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

    sender = TensorTransfer(host, port, tensor)
    receiver = TensorTransfer(host, port)

    start_time = time.time()

    sender.start()
    receiver.start()

    sender.join()
    receiver.join()

    end_time = time.time()

    print(f"Time taken: {end_time - start_time:.6f} seconds")
