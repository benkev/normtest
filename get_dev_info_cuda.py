# import pycuda.autoinit
import pycuda as cu

for devnum in range(cu.driver.Device.count()):
    dev = cu.driver.Device(devnum)
    attrs = dev.get_attributes()

    #Beyond this point is just pretty printing
    # print("\n===Attributes for device %d" % devnum)
    # for (key,val) in attrs.items():
    #     print("%s: %s" % (str(key),str(val)))


#
# Get attribute values from attrs dictionary
#
# Get dev and attrs for 0th GPU
#
dev = cu.driver.Device(0)
attrs = dev.get_attributes()

print()
print('From attrs dictionary:')
print("GLOBAL_MEMORY_BUS_WIDTH = %d" %
      attrs[cu.driver.device_attribute.GLOBAL_MEMORY_BUS_WIDTH])
print("CLOCK_RATE = %.3f MHz" %
      (attrs[cu.driver.device_attribute.CLOCK_RATE]/1e6))
print("MEMORY_CLOCK_RATE = %.3f MHz" %
      (attrs[cu.driver.device_attribute.MEMORY_CLOCK_RATE]/1e6))
print("TOTAL_CONSTANT_MEMORY = %.1f kB" %
      (attrs[cu.driver.device_attribute.TOTAL_CONSTANT_MEMORY]/2**10))
print("MAX_SHARED_MEMORY_PER_BLOCK = %.1f kB" %
      (attrs[cu.driver.device_attribute.MAX_SHARED_MEMORY_PER_BLOCK]/2**10))

#
# All device_attribute values may also be directly read as (lower-case)
# attributes on the Device object itself, e.g. dev.clock_rate.
#
print()
print('Directly as (lower-case) attributes on the Device object,')
print('  e.g. dev.clock_rate:')
print("global_memory_bus_width = %d" %
      dev.global_memory_bus_width)
print("clock_rate = %.3f mhz" %
      (dev.clock_rate/1e6))
print("memory_clock_rate = %.3f mhz" %
      (dev.memory_clock_rate/1e6))
print("total_constant_memory = %.1f kb" %
      (dev.total_constant_memory/2**10))
print("max_shared_memory_per_block = %.1f kb" %
      (dev.max_shared_memory_per_block/2**10))




#
# Return a tuple (free, total) indicating the free and total memory
# in the current context, in bytes.
#
(free,total)=cu.driver.mem_get_info()
print()
print("Global memory free:  %5.2f GB" % (free/2**30))
print("Global memory total: %5.2f GB" % (total/2**30))
print("Global memory occupancy: %f%% free" % (free*100/total))


