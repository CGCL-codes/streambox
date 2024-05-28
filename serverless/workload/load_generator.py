import os
import requests
import json
import numpy as np
import time
import yaml
from config import load_config, Config

class LoadGenerator:
    def __init__(self, config: Config):
        self.config = config.model_dump()
        self.server_url = f"http://{self.config['host']}:{self.config['port']}{self.config['path']}"
        print(f"POST requests to {self.server_url}")

    @staticmethod
    def get_workload(multipile: float, len: int, gap_time: int, lap_time: int, type: str):
        data, len_count = [], 0
        with open(f'./workload/{type}-load.txt', 'r') as file:
            numbers = file.readline().split()
        for number in numbers:
            data.append(int(int(number) * multipile))
            len_count += 1
            if len_count % gap_time == 0:
                data.extend([0]*lap_time)
            if len_count >= len:
                break
        return data

    def _post_request(self, data):
        try:
            response = requests.post(self.server_url, data=json.dumps(data), headers={'Content-Type': 'application/json'})
            response.raise_for_status()
        except (requests.HTTPError, Exception) as err:
            print(f"Error occurred: {err}")
            pass

    def generate_tasks(self, load_data):
        task_id_counter = 0
        for load in load_data:
            if load > 0:
                wait_times = np.random.exponential(1.0 / load, load)
                new_wait_times = wait_times / (np.sum(wait_times) / 1.0)
                for wait_time in new_wait_times:
                    task_data = {
                        "request_id": str(task_id_counter),
                        "model": "resnet50",
                        "data": {"vector": np.random.randint(1, 100, 4).tolist()}
                    }
                    self._post_request(task_data)
                    task_id_counter += 1
                    time.sleep(wait_time)
            time.sleep(max(1.0 - np.cumsum(new_wait_times)[-1], 0))
        return task_id_counter

# Usage example
config_path = os.getenv("CONFIG_FILE")
load_config = load_config(config_path, "load_generator")
load_gen = LoadGenerator(load_config)

workload = load_gen.get_workload(multipile=1, len=300, gap_time=20, lap_time=80, type='bursty')
load_gen.generate_tasks(workload)
