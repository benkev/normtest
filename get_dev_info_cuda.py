import pycuda.autoinit
import pycuda as cu

(free,total)=cu.driver.mem_get_info()
print("Global memory occupancy:%f%% free"%(free*100/total))

for devnum in range(cu.driver.Device.count()):
    dev = cu.driver.Device(devnum)
    attrs = dev.get_attributes()

    #Beyond this point is just pretty printing
    print("\n===Attributes for device %d" % devnum)
    for (key,val) in attrs.items():
        print("%s: %s" % (str(key),str(val)))


