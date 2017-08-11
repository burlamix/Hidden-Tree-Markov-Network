import numpy as np


t1 = np.ones((3,3))
t2 = np.zeros((3))
z=0
for i in range(0,3):
    for j in range(0,3):
            t1[i,j]=z
            t2[j]=z
            z=z+1

ris = np.divide(t1,t2)
print(t1)
print("______")
print(t2)
print("______")
print(ris)