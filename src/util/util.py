import time

G_START_TIME = time.time()

def TIMESTAMP(*args, type=None):
    if type == None:
        return 
    elapsed_time = round(time.time() - G_START_TIME, 10)
    print(f"[{elapsed_time}] ", end="")
    print(*args)

import inspect

def DEBUG(message):
    frame = inspect.currentframe().f_back
    filename = frame.f_code.co_filename
    lineno = frame.f_lineno
    function_name = frame.f_code.co_name
    
    print(f"[File: {filename}, Line: {lineno}, Function: {function_name}]", end="")
    print(message)