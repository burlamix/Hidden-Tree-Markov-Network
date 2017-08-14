import numpy as np
import tensorflow as tf
import tre_simple as ts
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

N_HIDDEN_STATES  = 4   #C
MAX_CHILD = 3          #L
N_TREE = 10             #NT
N_SYMBOLS = 3           #M
N_NODE = 100            #N
I_NODE = 70
MAX_LEVEL = 3

t = ts.List_tree(1)
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
in_prior = np.zeros((t.size,N_HIDDEN_STATES))


ph_pos_prior_p = tf.placeholder(shape=[N_HIDDEN_STATES,MAX_CHILD], dtype=tf.int32)
ph_sp_p = tf.placeholder(shape=[MAX_CHILD], dtype=tf.int32)
ph_pos_st_tr_p = tf.placeholder(shape=[N_HIDDEN_STATES,N_HIDDEN_STATES,MAX_CHILD], dtype=tf.int32)
ph_m_emission = tf.placeholder(shape=[N_HIDDEN_STATES,N_SYMBOLS], dtype=tf.int32)
ph_up_ward = tf.placeholder(shape=[N_NODE,N_HIDDEN_STATES], dtype=tf.int32)
ph_a_up_ward = tf.placeholder(shape=[MAX_CHILD,N_HIDDEN_STATES,N_NODE], dtype=tf.int32)
ph_post = tf.placeholder(shape=[N_NODE,N_HIDDEN_STATES], dtype=tf.int32)
ph_s_post = tf.placeholder(shape=[MAX_CHILD,N_HIDDEN_STATES,N_HIDDEN_STATES,N_NODE], dtype=tf.int32)
ph_in_prior = tf.placeholder(shape=[t.size,N_HIDDEN_STATES], dtype=tf.int32)

#___row test______________
z=100
h=0
for i in range(0,N_HIDDEN_STATES):
    for j in range(0, N_SYMBOLS):
        m_emission[i,j]=z
        h=h+1
        z=z+1
for i in range(0,N_HIDDEN_STATES ):
    for j in range(0, MAX_CHILD):
        pos_prior_p[i,j]=h
        h=h+1
w=0
for i in range(0,t.size):
    for j in range(0, N_HIDDEN_STATES):
        in_prior[i,j]=w
        w=w+1

for i in range(0,MAX_CHILD):
    sp_p[i]=i
for i in range(0,N_HIDDEN_STATES ):
    for j in range(0, N_HIDDEN_STATES):
        for k in range(0, MAX_CHILD):
            pos_st_tr_p[i,j,k]=10

#____________17_______________
print(in_prior)
aux1=tf.ones([N_HIDDEN_STATES, 0], tf.int32)
aux2=tf.ones([N_HIDDEN_STATES, 0], tf.int32)

for i in range(1,len(t.struct[-1])):
    a1 = tf.slice(ph_m_emission, [0, t.t.get_label(i)], [N_HIDDEN_STATES, 1])
    a2 = tf.slice(ph_pos_prior_p, [0, t.t.pos(i)], [N_HIDDEN_STATES, 1])
    aux1 = tf.concat([aux1,a1],1)
    aux2 = tf.concat([aux2,a2],1)


nume = tf.multiply(aux1,aux2)                              #Element-wise multiplication
den = tf.einsum('ij,ji->i', tf.transpose(aux1) ,aux2)      #Einstein summation per moltiplicazione di righe e colonne con lo stesso indice

ris_17_t = tf.divide(nume,den)             #17

#____________19 / 20___________

aux3 =tf.multiply(ph_sp_p,pos_st_tr_p)

childred=tf.ones([0,N_HIDDEN_STATES], tf.int32)

#for ma lo facciamo per un nodo solo per ora
aa =t.struct[-2][0]

print(t.struct)
print(aa)
print(aa.children)

for i in range(0,len(aa.children)):
    print(t.t.get_node(aa.children[i].name).name)
    child = tf.slice(ph_in_prior, [t.t.get_node(aa.children[i].name).name,0], [1,N_HIDDEN_STATES])
    childred = tf.concat([childred,child],0)

if __name__ == '__main__':
    aux4 =tf.multiply(aux3,tf.transpose(childred)) # bisogna verificare se il broodcast viene fatto correttamente...
#ora bisogna fare il broadcast esplicito per poi farne uno implicito per moltiplicare ogni matrice del nodo
#------------------------
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer(),)
    print(sess.run([aux4], {ph_m_emission: m_emission, ph_pos_prior_p: pos_prior_p,ph_sp_p: sp_p, ph_in_prior: in_prior}))








