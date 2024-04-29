"""
An implementation of Cannon's Algorithm for multiplyiing matrices
"""

from mpi4py import MPI
import numpy as np
import random
import math
from math import sqrt
# import time

# Initializing Communicator, size and rank
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

def matmul(A,B):
    result = [[sum(a * b for a, b in zip(A_row, B_col)) for B_col in zip(*B)] for A_row in A]
    return result



if(size!=4 and size!=16):
    print("Program doesn't support the input processor count. Select 4 or 16 processors")
    exit(1)


n = 4

nrows, ncols = int(sqrt(size)), int(sqrt(size))
dims = (nrows, ncols)
periods = (True, True)
reorder = False

# Defining the Cartesian system
crt2d = comm.Create_cart(dims, periods, reorder)
local_row, local_col = crt2d.Get_coords(rank)


# Initializing the adjacent procs based on the cartesian coordinates
dir = 1
d = 1
local_left, local_right = crt2d.Shift(direction = dir, disp=d)  

dir = 0
local_up, local_down = crt2d.Shift(direction = dir, disp = d)

# print(f"Rank:{rank}, left, right, up, down = ({local_left, local_right, local_up, local_down})")
# print(f"Rank:{rank}, loc = {local_row, local_col}")


# Initializing A and B matrices
if(rank==0):
    a = np.array([[ i for i in range(j*n+1,j*n+n+1) ] for j in range(n)], dtype=np.int32)
    print("A:\n", a)
    b = np.array([[ i+16 for i in range(j*n+1, j*n+n+1)] for j in range(n)], dtype=np.int32)
    print("B:\n", b)
else:
    a = None
    b = None


# Sub-matrix
sizes = (n, n)
subsizes = ( int(n/sqrt(size)), int(n/sqrt(size)) ) 
starts = (0,0)
order = MPI.ORDER_C

new_vect = MPI.INT.Create_subarray(sizes, subsizes, starts, order = order)
new_vect.Commit()


# Sending sub-matrices to the respective processes
if rank==0:
    for i in range(1, size):
        send_row, send_col = crt2d.Get_coords(i)
        comm.Send([( np.frombuffer( a.data, np.int32, offset=np.dtype(np.int32).itemsize * (send_row*n + send_col) * int(n/sqrt(size)) ) ), 1, new_vect], dest=i)
        comm.Send([( np.frombuffer( b.data, np.int32, offset=np.dtype(np.int32).itemsize * (send_row*n + send_col) * int(n/sqrt(size)) ) ), 1, new_vect], dest=i)
    
    a_local = np.ascontiguousarray(a[0:subsizes[0],0:subsizes[1]], dtype=np.int32)
    b_local = np.ascontiguousarray(b[0:subsizes[0],0:subsizes[1]], dtype=np.int32)
else:
    a_local = np.empty(subsizes, dtype=np.int32)
    b_local = np.empty(subsizes, dtype=np.int32)
    comm.Recv(a_local, source=0)
    comm.Recv(b_local, source=0)

# if rank==0:
# for i in range(size):
#     if(i==rank):
#         print(f"Rank:{rank}\n",a_local, "\n", b_local, "\n")
#     time.sleep(0.5)


iter = 0
C = np.zeros(subsizes, dtype=np.int32)

for iter in range(nrows):

    # if(iter==0):
    #     for j in range(n):
    #         if(local_row>j):
    #             comm.Sendrecv_replace(a_local, dest=local_left)

    if(iter<local_row):
        comm.send(obj=a_local, dest=local_left)
        new_a = comm.recv(source=local_right)
        a_local = new_a
    
    if(iter<local_col):
        comm.send(obj=b_local, dest=local_up)
        new_b = comm.recv(source=local_down)
        b_local = new_b

# for i in range(size):
#     if(i==rank):
#         print(f"Rank:{rank}\n",a_local, "\n", b_local, "\n")
#     time.sleep(0.5)

for iter in range(int(sqrt(size))):
    C_new = matmul(a_local, b_local)
    C = C + np.asarray(C_new)

    comm.send(obj=a_local, dest=local_left)
    new_a = comm.recv(source=local_right)
    a_local = new_a
    
    comm.send(obj=b_local, dest=local_up)
    new_b = comm.recv(source=local_down)
    b_local = new_b


# for i in range(size):
#     if(i==rank):
#         # print(f"Rank:{rank}\n",a_local, "\n", b_local, "\n")
#         print(f"Rank:{rank}\n",C)
#     time.sleep(0.5)


glob_C = np.zeros(sizes, dtype=np.int32)
C_dummy = np.zeros(sizes, dtype=np.int32)

for i in range(subsizes[0]):
    for j in range(subsizes[1]):
        C_dummy[local_row*subsizes[0]+i, local_col*subsizes[1]+j] = C[i,j]


# for iter_ in range(size):
#     if(rank==iter_):
#         print(f"Dummy C Gathered:\n {C_dummy}\n")
#         time.sleep(0.5) 

# The below statement is to Allgather all the C values from each proc
comm.Allreduce([C_dummy, MPI.INT], [glob_C, MPI.INT], op=MPI.SUM)

if(rank==0):
    print("C matrix from Cannon's Algorithm:\n", glob_C)
    # time.sleep(0.5)
    print(f"Matrix Multiplication:\n {matmul(a,b)}")


new_vect.Free()
MPI.Finalize()
