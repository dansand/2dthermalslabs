
# coding: utf-8

# In[ ]:




# In[2]:

from mpi4py import MPI
import numpy as np


# In[ ]:

comm = MPI.COMM_WORLD
rank = comm.rank


# In[3]:

sendbuf=[]
root=0
if comm.rank==0:
    m=np.array(range(comm.size*comm.size),dtype=float)
    m.shape=(comm.size,comm.size)
    print(m)
    sendbuf=m
v=comm.scatter(sendbuf,root)
print("I got this array:")
print(v)


# In[6]:

print("before barrier")
comm.barrier()
print("after barrier")


# sendbuf=[]
# root=0
# if comm.rank==0:
#     m=np.array([1])
#     #m.shape=(comm.size,comm.size)
#     print(m)
#     m.shape=(comm.size,comm.size)
#     sendbuf=m
# v=comm.scatter(sendbuf,root)
# print("I got this array:")
# print(v)

# In[ ]:




if rank == 0:
    #data = {'a':1,'b':2,'c':3}
    data = 1.
else:
    data = None

data = comm.bcast(data, root=0)
print 'rank',rank,data

