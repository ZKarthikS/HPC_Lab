"""
Odd-Even Sort parallelly

Split domain into p processors and sort using odd-even separately
Perform odd-even sort on these 'lists' to merge and sort
"""

#Output
"""
Time taken by the program for (1e4) random numbers:
np = 1: 12.914493004
np = 2: 3.559011068
np = 4: 0.91594711
np = 8: 0.340405466

For 10^5 random numbers:
np = 8: 25.88543912
"""

from mpi4py import MPI
import numpy as np
import math
import random
import time

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

def odd_even(list_):
    # Performs odd-even sort for the provided list_
    for i in range(len(list_)):
        for j in range(len(list_)-1):
            if((i+j)%2==0):
                if(list_[j]>list_[j+1]):
                    list_[j], list_[j+1] = list_[j+1], list_[j]

    return 

def sort_merge(list1, list2):
    # Takes two sorted lists - list1, list2
    # Provides a sorted [list1.items() + list2.items()]
    i, j = 0, 0
    sorted_list = []
    
    while i<len(list1) and j<len(list2):
        if(list1[i]<list2[j]):
            sorted_list.append(list1[i])
            i+=1
        else:
            sorted_list.append(list2[j])
            j+=1
    
    if(j==len(list2)):
        while i<len(list1):
            sorted_list.append(list1[i])
            i+=1
    
    
    if(i==len(list1)):
        while j<len(list2):
            sorted_list.append(list2[j])
            j+=1
    
    list1 = sorted_list[:i]
    list2 = sorted_list[i:]

    # print(sorted_list)

    return(list1,list2)

def split(list_, procs):
    new_list = []
    n = len(list_)
    for i in range(procs):
        newer_list = list_[int(i*n/procs):int((i+1)*n/size)]
        new_list.append(newer_list)
    
    return new_list


n = 10**4

start_time = MPI.Wtime()

if(rank==0):
    a = [(i) for i in range(n)]
    random.shuffle(a)

    proc_list = split(a,size)
    # print(f"List of items before iterations: {proc_list}")
else:
    proc_list = None

proc_list = comm.scatter(proc_list, root=0)

odd_even(proc_list)

for i in range(size):
    # print(f"List at rank={rank} after {i} iteration(s)")
    for j in range(size-1):
        if(i+j)%2==0:
            if(rank==j):
                new_list = comm.recv(source=rank+1, tag=0)
                # print(new_list)
                # print(rank, proc_list, new_list)
                proc_list, new_list = sort_merge(proc_list, new_list)
                # print(proc_list, new_list)
                comm.send(new_list, dest=rank+1, tag=1)
            elif rank==j+1:
                comm.send(proc_list, dest=rank-1, tag=0)
                proc_list = comm.recv(source=rank-1, tag=1)

proc_list = comm.gather(proc_list, root=0)

if(rank==0):
    a = []
    for i in range(size):
        a.extend(proc_list[i])
    # print(f"List reoccupied at rank=0 after all iterations: {a}")

end_time = MPI.Wtime()

if(rank==0):
    print(f"Time taken by the program = {end_time-start_time}")