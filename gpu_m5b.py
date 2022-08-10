import importlib
import sys

class gpu_m5b:

    name = 'gpu_m5b'

    #
    # Determine if PyCUDA or/and PyOpenCL are installed
    #
    pycuda_installed = importlib.util.find_spec("pycuda") is not None
    pyopencl_installed = importlib.util.find_spec("pyopencl") is not None

    #
    # In case both are installed, prefer PyCUDA since it is ~1.5 times faster
    #
    use_pycyda = False
    use_pyopencl = False
    if pycuda_installed:
        use_pycyda = True     # Use PyCUDA even if PyOpenCL installed
    elif pyopencl_installed:
        use_pyopencl = True   # Use PyOpenCL if PyCUDA not installed
    else:
        sys.exit("Neither PyCUDA nor PyOpenCL are installed. Exiting.")


    if use_pycyda:
        import pycuda as cu
        import pycuda.autoinit
        
        dev = cu.driver.Device(0)
        

        
    
    def __init__(self, a, b):
        self.a = a
        self.b = b 
    
    @classmethod
    def info(cls):
        return cls.name
