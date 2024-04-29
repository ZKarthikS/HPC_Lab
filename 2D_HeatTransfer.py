from mpi4py import MPI
import numpy as np
import random
import math
# import time

# Initializing Communicator, size and rank
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

t1 = MPI.Wtime()

nx = 40
ny = 40

x_dom_left = 0.0
x_dom_right = 1.0
y_dom_left = 0.0
y_dom_right = 1.0

delx = (x_dom_right-x_dom_left)/(nx-1)
dely = (y_dom_right-y_dom_left)/(ny-1)

nrows, ncols = 0, 0 # nrows and ncols are dimensions of the 2D proc grid
if(rank==0):
    (nrows, ncols) = map(int, input(f'Enter dimensions of {size} processor configuration\t').split())

    while(nrows*ncols != size):
        (nrows, ncols) = map(int, input(f'Invalid. Product of nrows and ncols = {size}\t').split())

# comm.Barrier()
nrows = comm.bcast(nrows, root=0)
ncols = comm.bcast(ncols, root=0)

dims = (nrows, ncols)
periods = (False, False)
reorder = False

crt2d = comm.Create_cart(dims, periods, reorder)
local_row, local_col = crt2d.Get_coords(rank)

dir = 1
d = 1
local_left, local_right = crt2d.Shift(direction = dir, disp=d)  

dir = 0
local_down, local_up = crt2d.Shift(direction = dir, disp = d)

# print(f"Rank:{rank}, left, right, up, down = ({local_left, local_right, local_up, local_down})")
# print(f"Rank:{rank}, loc = {local_row, local_col}")

# print(f'Rank={rank}, R,C = [{local_row}, {local_col}]')

local_nx = int((local_row+1)*nx/nrows) - int(local_row*nx/nrows)
local_ny = int((local_col+1)*ny/ncols) - int(local_col*ny/ncols)
loc_matrix = np.zeros((local_nx, local_ny), dtype=np.float64)

right_ext = np.zeros((local_nx), dtype=np.float64)
left_ext = np.zeros((local_nx), dtype=np.float64)
top_ext = np.zeros((local_ny), dtype=np.float64)
bottom_ext = np.zeros((local_ny), dtype=np.float64)


for i in range(local_nx):
    if local_left < 0:
        loc_matrix[i,0] = 400
    
    if local_right < 0:
        loc_matrix[i,-1] = 800

for i in range(local_ny):
    if local_up < 0:
        loc_matrix[-1,i] = 900
    
    if local_down < 0:
        loc_matrix[0,i] = 600


iter = 0
norm_err = 0.0
tol = 1e-4

matrix_new = loc_matrix.copy()

while(1):

    norm_err = 0.0
    iter += 1

    for i in range(1, local_nx-1):
        for j in range(1, local_ny-1):
            matrix_new[i,j] = (loc_matrix[i,j-1] + loc_matrix[i,j+1] + loc_matrix[i+1,j] + loc_matrix[i-1,j])/4
    

    i=0
    if local_down>=0:
        for j in range(1,local_ny-1):
            matrix_new[i,j] = (loc_matrix[i,j-1] + loc_matrix[i,j+1] + loc_matrix[i+1,j] + bottom_ext[j])/4
        
        j=0
        if local_left>=0:
            matrix_new[i,j] = (left_ext[i] + loc_matrix[i,j+1] + loc_matrix[i+1,j] + bottom_ext[j])/4
            
        
        j=local_ny-1
        if local_right>=0:
            matrix_new[i,j] = (loc_matrix[i,j-1] + right_ext[i] + loc_matrix[i+1,j] + bottom_ext[j])/4
    

    i=local_nx-1
    if local_up>=0:
        for j in range(1,local_ny-1):
            matrix_new[i,j] = (loc_matrix[i,j-1] + loc_matrix[i,j+1] + top_ext[j] + loc_matrix[i-1,j])/4
        
        j=0
        if local_left>=0:
            matrix_new[i,j] = (left_ext[i] + loc_matrix[i,j+1] + top_ext[j] + loc_matrix[i-1,j])/4
        
        j=local_ny-1
        if local_right>=0:
            matrix_new[i,j] = (loc_matrix[i,j-1] + right_ext[i] + top_ext[j] + loc_matrix[i-1,j])/4
    


    j=0
    if local_left>=0:
        for i in range(1,local_nx-1):
            matrix_new[i,j] = (left_ext[i] + loc_matrix[i,j+1] + loc_matrix[i+1,j] + loc_matrix[i-1,j])/4
    

    j=local_ny-1
    if local_right>=0:
        for i in range(1, local_nx-1):
            matrix_new[i,j] = (loc_matrix[i,j-1] + right_ext[i] + loc_matrix[i+1,j] + loc_matrix[i-1,j])/4
    
    norm_err = np.linalg.norm(matrix_new-loc_matrix)
    # norm_err = np.sum(np.divide(matrix_new-loc_matrix, loc_matrix))
    glob_err = comm.allreduce(norm_err, op=MPI.SUM)
    if glob_err < tol:
        loc_matrix = matrix_new.copy()
        print(f"Norm Err = {norm_err}")
        break

    loc_matrix = matrix_new.copy()

    # if(iter > 100):
    #     print(f"Norm Err = {norm_err}")
    #     break


    # Send-Receives
    if local_up>=0:
        comm.send(loc_matrix[-1,:], dest=local_up)
        top_ext = comm.recv(source = local_up)
        # print(f"Rank={rank}, local_up = {local_up}\ntop_ext={top_ext}")
        # break
    
    if local_down>=0:
        comm.send(loc_matrix[0,:], dest=local_down)
        bottom_ext = comm.recv(source = local_down)
    
    if local_left>=0:
        comm.send(loc_matrix[:,0], dest=local_left)
        left_ext = comm.recv(source=local_left)
        # for j in range(local_ny): 
        #     local_left[0,j] = var[j]
    
    if local_right>=0:
        comm.send(loc_matrix[:,-1], dest=local_right)
        right_ext = comm.recv(source=local_right)
        # for j in range(local_ny):
        #     local_right[0,j] = var[j]

        

# for i in range(size):
#     if(i==rank):
#         print(f"Rank:{rank}, left, right, up, down = ({local_left, local_right, local_up, local_down})\nnx, ny: {local_nx, local_ny}\n")
#         print(loc_matrix)
#     time.sleep(0.5)


mat_dum = np.zeros((nx,ny), dtype=np.float64)

for i in range(local_nx):
    for j in range(local_ny):
        mat_dum[int((local_row)*nx/nrows)+i, int(local_col*ny/ncols)+j] = loc_matrix[i,j]

mat_dum = 100*mat_dum
mat_dum = (np.rint(mat_dum)).astype(int)
mat_dum = mat_dum.astype(float)
mat_dum = mat_dum/100

# for i in range(size):
#     if(i==rank):
#         # print(f"Rank:{rank}, left, right, up, down = ({local_left, local_right, local_up, local_down})\nnx, ny: {local_nx, local_ny}\n")
#         print(f"Rank:{rank}\nnx, ny: {local_nx, local_ny}\n")
#         print(mat_dum)
#     time.sleep(0.5)

glob_matrix = comm.allreduce(mat_dum, op=MPI.SUM)

if(rank==0):
    for i in range(nx):
        for j in range(ny):
            print(glob_matrix[i][j], end=" ")
        print()
    # print(f"Final Matrix\n{glob_matrix}")


t2 = MPI.Wtime()

if(rank==0):
    print(f"Time taken = {t2-t1}")


MPI.Finalize()
