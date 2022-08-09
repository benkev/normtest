import importlib

class gpu_m5b:

    name = 'gpu_m5b'
    pycuda_installed = False
    pyopencl_installed = False

    pycuda_installed = importlib.util.find_spec("pycuda") is not None
    pyopencl_installed = importlib.util.find_spec("pyopencl") is not None



    
    def __init__(self, a, b):
        self.a = a
        self.b = b 
    
    @classmethod
    def info(cls):
        return cls.name
