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
in_prior = np.zeros((N_HIDDEN_STATES,t.size))
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
for i in range(0,N_HIDDEN_STATES):
    for j in range(0,t.size ):
        in_prior[i,j]=w
        w=w+1

for i in range(0,MAX_CHILD):
    sp_p[i]=i+1

w=0
for i in range(0,N_HIDDEN_STATES ):
    for j in range(0, N_HIDDEN_STATES):
        for k in range(0, MAX_CHILD):
            pos_st_tr_p[i,j,k]=w
            w=w+3


ph_pos_prior_p = tf.placeholder(shape=[N_HIDDEN_STATES,MAX_CHILD], dtype=tf.int32)
ph_sp_p = tf.placeholder(shape=[MAX_CHILD], dtype=tf.int32)
ph_pos_st_tr_p = tf.placeholder(shape=[N_HIDDEN_STATES,N_HIDDEN_STATES,MAX_CHILD], dtype=tf.int32)
ph_m_emission = tf.placeholder(shape=[N_HIDDEN_STATES,N_SYMBOLS], dtype=tf.int32)
ph_up_ward = tf.placeholder(shape=[N_NODE,N_HIDDEN_STATES], dtype=tf.int32)
ph_a_up_ward = tf.placeholder(shape=[MAX_CHILD,N_HIDDEN_STATES,N_NODE], dtype=tf.int32)
ph_post = tf.placeholder(shape=[N_NODE,N_HIDDEN_STATES], dtype=tf.int32)
ph_s_post = tf.placeholder(shape=[MAX_CHILD,N_HIDDEN_STATES,N_HIDDEN_STATES,N_NODE], dtype=tf.int32)


init = tf.constant(in_prior, dtype=tf.int32)
var_in_prior = tf.get_variable('var', initializer=init)

print(var_in_prior)

#____________17_______________
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


aux3 =tf.multiply(ph_sp_p,ph_pos_st_tr_p) # broadcast implicito


#per ogni livello dell'albero
for i in range(len(t.struct)-2,0,-1):
    liv_molt = tf.ones([0, N_HIDDEN_STATES], tf.int32)
    b = []
    print("____________________")

    #per ogni nodo del livello
    for node in t.struct[i]:
        print(node.name)

        children = tf.ones([N_HIDDEN_STATES,0 ], tf.int32)

        #per ogni figlio del nodo del livello
        for child_node in node.children:
            print(var_in_prior)

            child = tf.slice(var_in_prior, [0,child_node.name], [N_HIDDEN_STATES,1 ])    #estraggo il node prior
            #child = in_prior[:,child_node.name]    #estraggo il node prior
            #child = np.expand_dims(child,axis=1)
            print(var_in_prior)

            children = tf.concat([children, child],1)                                   #creo una matrice dei node_prior per figli di un nodo

        children = tf.expand_dims(children, 0)                                          #faccio un broadcast esplicito duplicando la matrice su una nuova dimenzione  per il numero di stati nascosti
        children = tf.tile(children, [N_HIDDEN_STATES, 1, 1])
        b.append(children)                                                              #metto tutte le matrici in una cosa

    liv_molt = tf.stack(b, 0) #concateno tutte le matrici N_HIDDEN_STATES*L dei node_prior per ogni gruppo di figli di un nodo  in un unica matrice N_HIDDEN_STATES*L*(numero di nodi del livello)

    #qui moltiplicazione
    aux4 = tf.multiply(aux3,liv_molt)       # questa è una serie di matrici, tante quanti sono i nodi del livello esaminati

    s = tf.reduce_sum(aux4,[2,3])           #sommo le matrici di dim 2
    s = tf.transpose(s)
    print(t.struct[i][0].name,"-------",t.struct[i][-1].name+1)
   # print(in_prior[:, t.struct[i][0].name : t.struct[i][-1].name+1])
    #print(s)
    #in_prior[:, t.struct[i][0].name : t.struct[i][-1].name+1] = s.eval()
    #in_prior[1,5] = s[3,1]
    tt = var_in_prior[1,1] * var_in_prior[2,2]
    #var_in_prior[:, t.struct[i][0].name : t.struct[i][-1].name].assign(s)
#------------------------
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer(),)
    print(sess.run([var_in_prior,s,tt], {ph_m_emission: m_emission, ph_pos_prior_p: pos_prior_p,ph_sp_p: sp_p, ph_pos_st_tr_p: pos_st_tr_p}))








