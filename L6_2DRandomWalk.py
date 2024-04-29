"""
An implementation of the randomwalk problem in 2-dimensions
A cartesian topology is used to mark the processors.
"""

from mpi4py import MPI
import numpy as np
import random
import math
import csv


# Initializing Communicator, size and rank
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


# Initializing Common Variables
N = int(20)    # Number of particles = 1000
domain = 100    # Domain = 0 to 100
I = int(10)    # Number of iterations = 1000
step = 25       # Step size = +/- 25



# Ensuring number of procs is 4, 8, 12 or 16 only. The program doesn't work otherwise
if(size%4 != 0 or size>16):
    if(rank==0):
        print("Number of processors input is not suitable. Please select number of procs as 4 or 8 or 12 or 16")
    exit(1)



# Initializing the topology based on number of processors
if(size == 4):
    rows = 4   # No. of rows
    cols = 1   # No. of columns
    dims = (rows, cols) # Dimensions
else:
    rows = 4   # No. of rows
    cols = int(size/4)  # No. of Columns
    dims = (rows, cols) # Dimensions

periods = (True, True)  # Ensures periodicity on both x and y directions
reorder = False



# Displays the topology on the output screen for reference
if(rank==0):
    print("Topology:")
    for i in range(rows-1,-1,-1):
        for j in range(cols):
            print(f"Proc {j + i*cols}\t", end='')
        print("")

crt2d = comm.Create_cart(dims, periods, reorder)    # Creates the cartesian topology
local_row, local_col = crt2d.Get_coords(rank)   # Provides details of the row and column based on the rank

# print(f"Rank={rank}, local_row={local_row}, local_col={local_col}")



dir = 1     # Direction along x-axis
d = 1       # Displacement = 1 unit
local_left, local_right = crt2d.Shift(direction = dir, disp=d)  # Finding left and right procs of each processor

dir = 0     # Direction along y-axis
local_up, local_down = crt2d.Shift(direction = dir, disp = d)   # Finding the proc above and below each processor

# print(f"The neighbours of rank={rank} is left={local_left}, right={local_right}, up={local_up}, down={local_down}")



# Defining the domains for each process
x_domain_left = int(local_col*domain/cols)
x_domain_right = int((local_col+1)*domain/cols)

y_domain_down = int(local_row*domain/rows)
y_domain_up = int((local_row+1)*domain/rows)

# print(f"The domain of rank={rank} is x_domain=({x_domain_left, x_domain_right}, y_domain={y_domain_down, y_domain_up}")



# Generating the particles and assigning IDs and initial positions:
obj_list = {}

if(rank==0):
    # Below piece of code generates 'N' random particles in the domain ([0,100], [0,100])
    obj_list = {i: [round((random.uniform(0,domain)),4), round((random.uniform(0,domain)),4)] for i in range(N)}
    # print(obj_list)

obj_list = comm.bcast(obj_list, root=0) # All particles are broadcast to all processors

# Deleting particles that do not belong in the domain of each process
del_keys = []
for k,v in obj_list.items():
    if not x_domain_left <= v[0] <= x_domain_right:
        del_keys.append(k)
    elif not y_domain_down <= v[1] <= y_domain_up:
        del_keys.append(k)

for k in del_keys:
    del obj_list[k]

# After deleting, each process contains only those particles that fall under it's domain

print(f"Number of particles in rank={rank} before {I} iterations is {len(obj_list)}")
# print(f"Particles in rank={rank} before {I} iterations is {(obj_list)}")



# The following code is to create files for Particle ID 40 and to number of particles in Proc_0
if(rank == 0):
    f = open('Particle_40.csv', "w")
    writer = csv.writer(f)
    writer.writerow(['Iteration', 'x-Position', 'y-Position'])
    f.close()
    
    f_ = open('L6_Proc_0.csv', "w")
    writer = csv.writer(f_)
    writer.writerow(['No. of Particles'])
    f_.close()



# Assigning iteration variable to 0 and beginning the iterations
iter = 0

for iter in range(I):

    del_keys = []
    send_left = {}
    send_right = {}
    send_up = {}
    send_down = {}
    new_obj_1 = {}
    new_obj_2 = {}


    # Counting the number of particles in proc_0 at each iteration
    if(rank==0):
        f = open('L6_Proc_0.csv', "a")
        writer = csv.writer(f)
        writer.writerow([len(obj_list)])
        f.close()
    

    # This for loop creates a random step for each particle in the x and y direction in the range (-25,25)
    # It creats dicts to send and receive particles only in the x-direction (left-right) based on domain limits
    for k, v in obj_list.items():
        x_step = random.uniform(-1*step,step)
        y_step = random.uniform(-1*step,step)

        obj_list[k][0] = round(obj_list[k][0]+x_step,4)
        obj_list[k][1] = round(obj_list[k][1]+y_step,4)

        if v[0] < x_domain_left:
            # send_left[k] = [round(v[0]%domain, 4), round(v[1], 4)]
            send_left[k] = [round(v[0], 4), round(v[1], 4)]
            if local_left!=rank:
                del_keys.append(k)
            # del_keys.append(k)
        elif v[0] > x_domain_right:
            # send_right[k] = [round(v[0]%domain, 4), round(v[1], 4)]
            send_right[k] = [round(v[0], 4), round(v[1], 4)]
            if local_right!=rank:
                del_keys.append(k)
            # del_keys.append(k)
        
        # print(f" Rank = {rank}, left={send_left}, right={send_right}")
    

    # Send-Receive occurs in the x-direction
    comm.send(obj=send_left, dest=local_left, tag=0)
    new_obj_1 = comm.recv(source=local_right)

    comm.send(obj=send_right, dest=local_right, tag=0)
    new_obj_2 = comm.recv(source=local_left)
    
    
    # Adds the new received items to the processors's object list
    for k,v in new_obj_1.items():
        obj_list[k] = [round(v[0]%domain, 4), round(v[1]%domain, 4)]
    for k,v in new_obj_2.items():
        obj_list[k] = [round(v[0]%domain, 4), round(v[1]%domain, 4)]
    
    # if(local_left==rank):
    #     del_keys = []
    
    # Deletes the objects that were sent to other processors
    for k in del_keys:
        del obj_list[k]
    
    del_keys = []


    # This for loop creats dicts to send and receive particles only in the y-direction (above-below) based on domain limits
    for k,v in obj_list.items():
        if v[1] < y_domain_down:
            # send_down[k] = [v[0], round(v[1]%domain, 4)]
            send_down[k] = [round(v[0],4), round(v[1], 4)]
            if local_down!=rank:
                del_keys.append(k)
            # del_keys.append(k)
        elif v[1] > y_domain_up:
            send_up[k] = [round(v[0],4), round(v[1], 4)]
            if local_up!=rank:
                del_keys.append(k)
            # del_keys.append(k)
        
        if(k==40):
            f = open('Particle_40.csv', "a")
            writer = csv.writer(f)
            writer.writerow([iter, round(v[0]%domain, 4), round(v[1]%domain, 4)])
            f.close()
    

    # Send-Receive occurs in the y-direction
    comm.send(obj=send_down, dest=local_down, tag=0)
    new_obj_1 = comm.recv(source=local_up)

    comm.send(obj=send_up, dest=local_up, tag=0)
    new_obj_2 = comm.recv(source=local_down)
    
    
    # Adds the new received items to the processors's object list
    for k,v in new_obj_1.items():
        obj_list[k] = [round(v[0]%domain, 4), round(v[1]%domain, 4)]
    for k,v in new_obj_2.items():
        obj_list[k] = [round(v[0]%domain, 4), round(v[1]%domain, 4)]
    
    # print(f"Objects in rank={rank}: {obj_list}")
    # comm.Barrier()

    # if(local_down==rank):
    #     del_keys = []
    
    # Deletes the objects that were sent to other processors
    for k in del_keys:
        del obj_list[k]
    
    # print(f"Objects in rank={rank}: {obj_list}")

print(f"Objects in rank={rank}: {obj_list}")
print(f"Number of particles in rank={rank} after {I} iterations is {len(obj_list)}")

if(rank==0):    
    f_ = open('L6_all_particles.csv', "w")
    writer = csv.writer(f_)
    writer.writerow(['Particle ID', 'x-Position', 'y-Position'])
    f_.close()

for j in range(N):
    if j in obj_list.keys():
        f = open('L6_all_particles.csv', "a")
        writer = csv.writer(f)
        writer.writerow([j, obj_list[j][0], obj_list[j][1]])
        f.close()
    comm.Barrier()
