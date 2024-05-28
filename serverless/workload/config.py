from pydantic import BaseModel, Field, validator
import re

class Config(BaseModel):
    host: str = Field(default="localhost")
    port: int = Field(default=8080)
    path: str = Field(default="/api/batcher")

    @validator('port')
    def validate_port(cls, port):
        if not 1024 <= port <= 65535:
            raise ValueError('port must be in 1024-65535')
        return port

    @validator('path')
    def validate_path(cls, path):
        # Path must start with / and can contain multiple parts separated by / (each part can contain a-z, A-Z, 0-9, -, _)
        pattern = re.compile('^\/[a-zA-Z0-9\-_]+(\/[a-zA-Z0-9\-_]*)*$')
        if not pattern.match(path):
            raise ValueError('path must start with / and can contain multiple parts separated by / (each part can contain a-z, A-Z, 0-9, -, _)')
        return path


import yaml
from pydantic import ValidationError

def load_config(file_path: str, config_key: str):
    with open(file_path, 'r') as file:
        config_dict = yaml.safe_load(file)
    
    try:
        config = Config(**config_dict[config_key])
    except ValidationError as e:
        print(e.json(indent=2))
        raise e
    
    return config
