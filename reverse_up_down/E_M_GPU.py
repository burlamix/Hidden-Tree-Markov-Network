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
pos_prior_p = np.ones((N_HIDDEN_STATES,MAX_CHILD))
#SP probability matrix --- fi
sp_p = np.ones((MAX_CHILD))
#positional state transiction probability matrix --- A
pos_st_tr_p = np.ones((N_HIDDEN_STATES,N_HIDDEN_STATES,MAX_CHILD)) # assumo che gli inidici sono j, i, l
#multinomial emision --- bi
m_emission = np.ones((N_HIDDEN_STATES,N_SYMBOLS))

#upward parameters beta
up_ward = np.ones((N_NODE,N_HIDDEN_STATES))
a_up_ward = np.ones((MAX_CHILD,N_HIDDEN_STATES,N_NODE))

#stater posterior €
post = np.ones((N_NODE,N_HIDDEN_STATES))
#pairwwise smoothed posterior
s_post = np.ones((MAX_CHILD,N_HIDDEN_STATES,N_HIDDEN_STATES,N_NODE))


#internal node prior
in_prior = np.zeros((N_NODE,N_HIDDEN_STATES))


z=100
h=0
for i in range(0,N_HIDDEN_STATES):
    for j in range(0, N_SYMBOLS):
        m_emission[i,j]=z
        z=z+1
for i in range(0,N_HIDDEN_STATES ):
    for j in range(0, MAX_CHILD):
        pos_prior_p[i,j]=h
        h=h+1
print("\nm emission  bi")
print(m_emission)
print("\npos_prior_p  pi")
print(pos_prior_p)
print("\nalbero")
print(t.struct)
# estraggo e calcolo il numeratore e denominatore di tutti i risultati della 17
aux1 = np.ones((N_HIDDEN_STATES,len(t.struct[-1])))
aux2 = np.ones((N_HIDDEN_STATES,len(t.struct[-1])))

for i in range(1,len(t.struct[-1])):
    #aux1[:,i] = m_emission[:,i]
    #aux2[:,i] = pos_prior_p[:,i]
    aux1[:,i] = m_emission[:,int(t.t.get_label(i))]
    aux2[:,i] = pos_prior_p[:,t.t.pos(i)]

numerator = np.multiply(aux1,aux2)                              #Element-wise multiplication
denominator = np.einsum('ij,ji->i', np.transpose(aux1) ,aux2)   #Einstein summation per moltiplicazione di righe e colonne con lo stesso indice

ris_17 = np.divide(numerator,denominator)             #17


aux1 = np.ones((N_HIDDEN_STATES,MAX_CHILD,7))
for u in range(0,7):
    for i in range(0,MAX_CHILD):
        aux1[:,i,u]=in_prior[i,:]

print(sp_p.shape)
print(pos_st_tr_p.shape)
print(aux1.shape)


#da_somma = np.multiply(sp_p,pos_st_tr_p,aux1)
da_somma = sp_p*pos_st_tr_p*aux1



'''print("\n aux1")
print(aux1)
print("\n aux2")
print(aux2)
print("\n nmeratore")
print(numerator)
print("\n denominatore")
print(denominator)
print("\n risultato")
print(ris_17)'''



