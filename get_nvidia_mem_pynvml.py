from pynvml import *
nvmlInit()
for i in range(nvmlDeviceGetCount()):
    handle = nvmlDeviceGetHandleByIndex(i)
    meminfo = nvmlDeviceGetMemoryInfo(handle)
    print("%s: %0.1f MB free, %0.1f MB used, %0.1f MB total" % (
        # <class 'bytes'> --> <class 'str'>
        nvmlDeviceGetName(handle).decode('UTF-8'), 
        meminfo.free/1024.**2, meminfo.used/1024.**2, meminfo.total/1024.**2))
nvmlShutdown()
