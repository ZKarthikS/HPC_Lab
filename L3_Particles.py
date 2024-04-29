"""
Particle ID - 1 to 10000
Number of particles - 10000
Number of iterations - 1000

At the initial step, assign particles a random position from 0 to 100.
Divide the particles based on its position to p processors.
At each iteration, the particle can move one position forward or 1 
position backward rand([-1, 1]).
If any particle crosses a boundary, send/receive the particle to 
the next/previous processor.
"""

# Save number of particles in each proc at the end of each iteration
# Plot the graph of the same 

from mpi4py import MPI
import numpy as np
import random
import math
import csv

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

N = int(1e4)
domain = 100
I = int(1e3)

if(size<=1):
    print("Insufficient processors.")
    exit(1)

obj_list = {}

if(rank==0):
    
    # obj_list = []
    # proc_count =[0 for i in range(size)]

    obj_list = {i: round((random.uniform(0,domain)),4) for i in range(N)}

obj_list = comm.bcast(obj_list, root=0)

# Both domain ends are inclusive in that processor
proc_domain_start = int(rank*domain/size)
proc_domain_end = int((rank+1)*domain/size)

del_keys = []
for k,v in obj_list.items():
    if not proc_domain_start <= v <= proc_domain_end:
        del_keys.append(k)

for item in del_keys:
    del obj_list[item]

# print(f"Objects in rank={rank}: {obj_list}")
print(f"Number of particles in rank={rank} before {I} iterations is {len(obj_list)}")

iter = 0

comm.barrier()

if(rank == 0):
    f = open('Particle.csv', "w")
    writer = csv.writer(f)
    writer.writerow(['Iteration', 'Position'])
    f.close()


f_ = open(f'Proc{rank}.csv', "w")
writer = csv.writer(f_)
writer.writerow(['No. of Particles'])
f_.close()

for iter in range(I):

    del_keys = []
    gather_obj = {}

    f = open(f'Proc{rank}.csv', "a")
    writer = csv.writer(f)
    writer.writerow([len(obj_list)])
    f.close()
    
    for k,v in obj_list.items():
        # step = random.choice([-1,1])
        # print(step)
        step = random.uniform(-1,1)
        obj_list[k] = v + step

        if not proc_domain_start <= v <= proc_domain_end:
            del_keys.append(k)
            gather_obj[k] = round(v%domain, 4)
        
        if(k==40):
            # print(step, v)
            f = open('Particle.csv', "a")
            writer = csv.writer(f)
            writer.writerow([iter, round(v%domain, 4)])
            f.close()
    
    # print(f"Gather obj at rank={rank}: {gather_obj}")

    comm.barrier()
    comm.bcast(iter, root=0)
    gather_obj = comm.allgather(gather_obj)

    for j in gather_obj:
        for k,v in j.items():
            if proc_domain_start <= v <= proc_domain_end:
                obj_list[k] = v
    
    for item in del_keys:
        del obj_list[item]
    
    # print(f"Objects in rank={rank}: {obj_list}")
print(f"Number of particles in rank={rank} after {I} iterations is {len(obj_list)}")