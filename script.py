import multiprocessing as mp

def print_affinity():
    import time
    import psutil
    p = psutil.Process()
    print('before import torch', p.cpu_affinity())
p = mp.Process(target=print_affinity)
p.start()
p.join()

import torch

def print_affinity():
    import time
    import psutil
    p = psutil.Process()
    print('after import torch', p.cpu_affinity())
p = mp.Process(target=print_affinity)
p.start()
p.join()