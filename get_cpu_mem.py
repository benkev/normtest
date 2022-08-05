import os
import re

mem1 = os.popen('free -b').readlines()
mem1 = mem1[1].split()
tot_ram1 = float(mem1[1]) 
fre_ram1 = float(mem1[3]) 
avl_ram1 = float(mem1[6]) 

print()
print('Using os.popen(''free -b'').readlines():')
print('CPU RAM: total %5.2f GB, available %5.2f GB, free %5.2f GB' %
      (tot_ram1/2**30, avl_ram1/2**30, fre_ram1/2**30))

with open('/proc/meminfo') as f:
    mem2_tot = float(f.readline().split()[1])
    mem2_fre = float(f.readline().split()[1])
    mem2_avl = float(f.readline().split()[1])

   
# matched = re.search(r'^MemTotal:\s+(\d+)', meminfo)
# if matched: 
#     mem_total_kB = int(matched.groups()[0])


print()
print('Using /proc/meminfo:')
print('CPU RAM: total %5.2f GB, available %5.2f GB, free %5.2f GB' %
      (mem2_tot/2**20, mem2_avl/2**20, mem2_fre/2**20))

#
# Using psutil
#
import psutil

mem = psutil.virtual_memory() 

print()
print('Using cross-platform psutil.virtual_memory():')
print('total:     %5.2f GB' % (mem.total/2**30))
print('available: %5.2f GB' % (mem.available/2**30))
print('used:      %5.2f GB' % (mem.used/2**30))

# memd = mem._asdict()
# for nam, val in mem._asdict().items():
#     print('%s\t = %5.2f' % (nam, val))



