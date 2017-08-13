import numpy as np

import tre_simple as ts

N_HIDDEN_STATES  = 4   #C
MAX_CHILD = 3           #L
N_TREE = 10             #NT
N_SYMBOLS = 3           #M
N_NODE = 100            #N
I_NODE = 70
MAX_LEVEL = 2

import numpy as np

import tre_simple as ts

N_HIDDEN_STATES  = 4   #C
MAX_CHILD = 3           #L
N_TREE = 10             #NT
N_SYMBOLS = 3           #M
N_NODE = 100            #N
I_NODE = 70
MAX_LEVEL = 2

t = ts.List_tree("0")
t.t.make_linear_tree(MAX_CHILD, MAX_LEVEL, N_SYMBOLS)
t.divide_leaves()
t.set_name()

#model parameters
#positional prior probability matrix --- pi
a = np.ones((N_HIDDEN_STATES,MAX_CHILD))
a2 = np.ones((N_HIDDEN_STATES,MAX_CHILD))
#SP probability matrix --- fi
b = np.ones((MAX_CHILD))

z=1
h=10
for i in range(0,N_HIDDEN_STATES ):
    for j in range(0,N_SYMBOLS ):
        a[i,j]=z
        z=z+1

for i in range(0,N_HIDDEN_STATES ):
    for j in range(0,N_SYMBOLS ):
        a2[i,j]=2

for j in range(0, MAX_CHILD):
    b[j]=1
    h=h+1

print(a)
print(a2)
print(b)
#c = np.broadcast_to(b,(4,3))
#c = np.broadcast_arrays(b,a)

r = np.multiply(b,a,a2)


print(r)

