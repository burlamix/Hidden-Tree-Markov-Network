import numpy as np
import tensorflow as tf
from tre_simple import *
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'



N_HIDDEN_STATES  = 4   #C
MAX_CHILD = 3          #L
N_TREE = 10             #NT
N_SYMBOLS = 5           #M
I_NODE = 70
MAX_LEVEL = 3

t = List_tree(1)
t.t.make_linear_tree(MAX_CHILD, MAX_LEVEL, N_SYMBOLS)
t.divide_leaves()
t.set_name()

#model parameters
#positional prior probability matrix --- pi
pi = np.ones((N_HIDDEN_STATES,MAX_CHILD))
#SP probability matrix --- fi
sp_p = np.ones((MAX_CHILD))
#positional state transiction probability matrix --- A
A = np.ones((N_HIDDEN_STATES,N_HIDDEN_STATES,MAX_CHILD)) # assumo che gli inidici sono j, i, l
#multinomial emision --- bi
bi = np.ones((N_HIDDEN_STATES,N_SYMBOLS))

#upward parameters beta
up_ward = np.ones((t.size,N_HIDDEN_STATES))
a_up_ward = np.ones((t.size,N_HIDDEN_STATES,MAX_CHILD))

#stater posterior €
post = np.ones((t.size,N_HIDDEN_STATES))
#pairwwise smoothed posterior
s_post = np.ones((MAX_CHILD,N_HIDDEN_STATES,N_HIDDEN_STATES,t.size))


#internal node prior
in_prior = np.zeros((N_HIDDEN_STATES,t.size))
in_prior2 = np.zeros((N_HIDDEN_STATES,t.size))


#pairwise smoothed posterior
E = np.zeros((N_HIDDEN_STATES,t.size))
EE = np.zeros((t.size,N_HIDDEN_STATES,N_HIDDEN_STATES))



#___row test______________
z=100
h=0
for i in range(0,N_HIDDEN_STATES):
    for j in range(0, N_SYMBOLS):
        bi[i,j]=z
        h=h+1
        z=z+1
for i in range(0,N_HIDDEN_STATES ):
    for j in range(0, MAX_CHILD):
        pi[i,j]=h
        h=h+1
w=1
for i in range(0,N_HIDDEN_STATES):
    for j in range(t.size-len(t.struct[-1]),t.size ):
        in_prior[i,j]=w
        w=w+1

for i in range(0,MAX_CHILD):
    sp_p[i]=i+1

w=0
for i in range(0,N_HIDDEN_STATES ):
    for j in range(0, N_HIDDEN_STATES):
        for k in range(0, MAX_CHILD):
            A[i,j,k]=w
            w=w+3
z=1
for i in range(0,t.size):
    for j in range(0, N_HIDDEN_STATES):
        up_ward[i,j]=z
        z=z+1


w=0
for i in range(0,t.size ):
    for j in range(0, N_HIDDEN_STATES):
        for k in range(0, MAX_CHILD):
            a_up_ward[i,j,k]=w
            w=w+1

z=0
for j in range(0, t.size ):
    for i in range(0,N_HIDDEN_STATES):
        E[i,j]=0
        z=z+1

ph_pi = tf.placeholder(shape=[N_HIDDEN_STATES,MAX_CHILD], dtype=tf.float64)
ph_sp_p = tf.placeholder(shape=[MAX_CHILD], dtype=tf.float64)
ph_A = tf.placeholder(shape=[N_HIDDEN_STATES,N_HIDDEN_STATES,MAX_CHILD], dtype=tf.float64)
ph_bi = tf.placeholder(shape=[N_HIDDEN_STATES,N_SYMBOLS], dtype=tf.float64)
ph_post = tf.placeholder(shape=[t.size,N_HIDDEN_STATES], dtype=tf.float64)
ph_s_post = tf.placeholder(shape=[MAX_CHILD,N_HIDDEN_STATES,N_HIDDEN_STATES,t.size], dtype=tf.float64)
ph_in_prior = tf.placeholder(shape=[N_HIDDEN_STATES,t.size], dtype=tf.float64)


init_prior = tf.constant(in_prior, dtype=tf.float64)
var_in_prior = tf.get_variable('var_in_prior', initializer=init_prior)

init_a_up_ward = tf.constant(a_up_ward, dtype=tf.float64)
var_a_up_ward = tf.get_variable('var_a_up_ward', initializer=init_a_up_ward)


init_up_ward = tf.constant(up_ward, dtype=tf.float64)
var_up_ward = tf.get_variable('var_up_ward', initializer=init_up_ward)


init_E = tf.constant(E, dtype=tf.float64)
var_E = tf.get_variable('E', initializer=init_E)

init_EE = tf.constant(EE, dtype=tf.float64)
var_EE = tf.get_variable('EE', initializer=init_EE)


#____________17_______________
aux1=tf.ones([N_HIDDEN_STATES, 0], tf.float64)
aux2=tf.ones([N_HIDDEN_STATES, 0], tf.float64)

for i in range(t.struct[-1][0].name,t.size):                #inefficente se il numero di nodi è elevato rispeto a le posizioni e ai label

    a1 = tf.slice(ph_bi, [0, t.t.get_label(i)], [N_HIDDEN_STATES, 1])
    a2 = tf.slice(ph_pi, [0, t.t.pos(i)], [N_HIDDEN_STATES, 1])
    aux1 = tf.concat([aux1,a1],1)
    aux2 = tf.concat([aux2,a2],1)


nume = tf.multiply(aux1,aux2)                              #Element-wise multiplication
den = tf.einsum('ij,ji->i', tf.transpose(aux1) ,aux2)      #Einstein summation per moltiplicazione di righe e colonne con lo stesso indice

ris_17_t = tf.divide(nume,den)             #17
ris_17_t = tf.transpose(ris_17_t, perm=[1, 0])

head = tf.slice(var_up_ward, [0, 0],[t.struct[-1][0].name, N_HIDDEN_STATES])
var_up_ward = tf.concat([head, ris_17_t], 0)



#____________ 20___________


aux3 =tf.multiply(ph_sp_p,ph_A) # broadcast implicito


#per ogni livello dell'albero

resto = len(t.struct[-1]) # numero foglie

for i in range(len(t.struct)-2,-1,-1):
    aux5 = tf.ones([0, N_HIDDEN_STATES], tf.float64)
    aux6 = []

    #per ogni nodo del livello
    for node in t.struct[i]:

        children = tf.ones([N_HIDDEN_STATES,0 ], tf.float64)

        #per ogni figlio del nodo del livello
        for child_node in node.children:

            child = tf.slice(var_in_prior, [0,child_node.name], [N_HIDDEN_STATES,1 ])    #estraggo il node prior
            children = tf.concat([children, child],1)                                    #creo una matrice dei node_prior per figli di un nodo

        children = tf.expand_dims(children, 0)                                          #faccio un broadcast esplicito duplicando la matrice su una nuova dimenzione  per il numero di stati nascosti
        children = tf.tile(children, [N_HIDDEN_STATES, 1, 1])
        aux6.append(children)

        aux5 = tf.stack(aux6, 0) #concateno tutte le matrici N_HIDDEN_STATES*L dei node_prior per ogni gruppo di figli di un nodo  in un unica matrice N_HIDDEN_STATES*L*(numero di nodi del livello)

    #qui moltiplicazione
    aux4 = tf.multiply(aux3,aux5)       # questa è una serie di matrici, tante quanti sono i nodi del livello esaminati

    s = tf.reduce_sum(aux4,[2,3])           #sommo le matrici di dim 2
    s = tf.transpose(s)

    head = tf.slice(var_in_prior, [0, 0], [N_HIDDEN_STATES, t.struct[i][0].name])                                       # potrei farlo anche con un constant
    tail = tf.slice(var_in_prior, [0, t.struct[i][-1].name+1], [N_HIDDEN_STATES, t.size - t.struct[i][-1].name-1])   # potrei farlo anche con un constant


    var_in_prior = tf.concat([head,s,tail],1)       #aggiorno i nuovi valore trovati

#____________ 21____19_______


# qui ci va un for per farlo a livelli dove estraggo da up_ward i vari nodi del livello

for i in range(len(t.struct) - 2, -1, -1):

#||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
    aux5 = tf.ones([0, N_HIDDEN_STATES], tf.float64)
    aux6 = []
    #per ogni nodo del livello
    node_in_priors = tf.ones([N_HIDDEN_STATES, 0], tf.float64)
    for node in t.struct[i]:

        children = tf.ones([0, N_HIDDEN_STATES], tf.float64)

        # per ogni figlio del nodo del livello
        for child_node in node.children:
            child = tf.slice(var_up_ward, [child_node.name,0 ], [1, N_HIDDEN_STATES])
            children = tf.concat([children, child], 0)

        children = tf.expand_dims(children, 0)  # faccio un broadcast esplicito duplicando la matrice su una nuova dimenzione  per il numero di stati nascosti
        children = tf.tile(children, [N_HIDDEN_STATES, 1, 1])
        children = tf.transpose(children, perm=[2,0,1])

        aux6.append(children)
        aux5 = tf.stack(aux6, 0)
        #print(node.name)
        node_in_prior = tf.slice(var_in_prior, [0, node.name], [N_HIDDEN_STATES, 1])
        node_in_priors = tf.concat([node_in_priors, node_in_prior], 1)


    node_in_priors = tf.expand_dims(node_in_priors, 0)  # faccio un broadcast esplicito duplicando la matrice su una nuova dimenzione  per il numero di stati nascosti
    node_in_priors = tf.tile(node_in_priors, [MAX_CHILD, 1, 1])
    node_in_priors = tf.transpose(node_in_priors, perm=[2, 1, 0])

    denominator_n = tf.multiply(ph_A, aux5)
    denominator = tf.reduce_sum(denominator_n, [1])  # sommo le matrici di dim 2
    s = tf.divide( denominator,node_in_priors,)



    head = tf.slice(var_a_up_ward, [0, 0, 0], [t.struct[i][0].name , N_HIDDEN_STATES, MAX_CHILD])                                       # potrei farlo anche con un constant
    tail = tf.slice(var_a_up_ward, [t.struct[i][-1].name+1,0,0], [t.size - t.struct[i][-1].name-1,N_HIDDEN_STATES, MAX_CHILD])          # potrei farlo anche con un constant

    var_a_up_ward = tf.concat([head,s,tail],0)

#|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||

    b=tf.ones([N_HIDDEN_STATES, 0], tf.float64)
    second_term=tf.ones([0, N_HIDDEN_STATES,MAX_CHILD], tf.float64)
    aux3=tf.ones([N_HIDDEN_STATES,0 ], tf.float64)

    for node in t.struct[i]:
        print("°°°°")
        a1 = tf.slice(ph_bi, [0, t.t.get_label(node.name)], [N_HIDDEN_STATES, 1])
        b = tf.concat([b,a1],1)

        a2 = tf.slice(var_a_up_ward,[node.name,0,0],[1,N_HIDDEN_STATES,MAX_CHILD])
        second_term = tf.concat([second_term,a2],0)

        a3 = tf.slice(var_in_prior,[0,node.name],[N_HIDDEN_STATES,1])
        aux3 = tf.concat([aux3,a3],1)



    first_term = tf.expand_dims(ph_sp_p, 0)  # faccio un broadcast esplicito duplicando la matrice su una nuova dimenzione  per il numero di stati nascosti
    first_term = tf.expand_dims(first_term, 0)
    first_term = tf.tile(first_term, [len(t.struct[i]),N_HIDDEN_STATES, 1])

    third_term = tf.expand_dims(aux3, 0)
    third_term = tf.tile(third_term, [MAX_CHILD,1, 1])
    third_term = tf.transpose(third_term, perm=[2, 1, 0])

    # per in num
    aux4 = tf.multiply(first_term, second_term)
    aux5 = tf.multiply(aux4,third_term)

    somm = tf.reduce_sum(aux5, [2])  # sommo sulla dim 2 (sommo le righe)

    b = tf.transpose(b, perm=[1, 0])

    numerator = tf.multiply(b,somm)

    # per il den
    bb = tf.expand_dims(b, 2)
    bb = tf.tile(bb, [1,1,MAX_CHILD])

    aux7 = tf.multiply(bb, first_term)
    aux8 = tf.multiply(aux7, second_term)
    aux9 = tf.multiply(aux8,third_term)

    denominator_int = tf.reduce_sum(aux9, [2,1])  # sommo sulla dim 2 (sommo le righe)

    denominator = tf.expand_dims(denominator_int, 1)
    denominator = tf.tile(denominator, [1, N_HIDDEN_STATES])    # questo è il Nu uno per ogni nodo, va duplicato per ogni stato nascosto di un nodo
    #finale

    ris_19 = tf.divide(numerator,denominator)

    head = tf.slice(var_up_ward, [0, 0], [t.struct[i][0].name, N_HIDDEN_STATES])
    tail = tf.slice(var_up_ward, [t.struct[i+1][0].name, 0], [t.size- t.struct[i+1][0].name, N_HIDDEN_STATES])

    var_up_ward = tf.concat([head, ris_19,tail], 0)
#|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
#|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
#|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
# down step



base = tf.slice(var_up_ward, [0, 0], [1, N_HIDDEN_STATES])
head = tf.slice(var_E,[0,1],[N_HIDDEN_STATES,t.size-1])
head = tf.transpose(head, perm=[1, 0])
var_E = tf.concat([base,head],0)

for i in range(1, len(t.struct) ):
    print("----------------------------------- "+str(i)+" ----------------------------------------------------------------")
    print(t.struct[i])

    sli_E = tf.zeros([0,N_HIDDEN_STATES], tf.float64)
    sli_in_prior = tf.zeros([N_HIDDEN_STATES,0], tf.float64)
    sli_up_ward = tf.zeros([0,N_HIDDEN_STATES], tf.float64)
    sli_sp_p = tf.zeros([0], tf.float64)
    sli_A = tf.zeros([0,N_HIDDEN_STATES,N_HIDDEN_STATES], tf.float64)
    sli_var_a_up_ward = tf.zeros([0,N_HIDDEN_STATES,MAX_CHILD], tf.float64)

    for node in t.struct[i]:
        # numer
        a_sli_E = tf.slice(var_E, [ t.t.pa(node.name).name,0], [1,N_HIDDEN_STATES])
        sli_E = tf.concat([sli_E,a_sli_E],0)

        a_sli_up_ward = tf.slice(var_up_ward, [node.name, 0], [1, N_HIDDEN_STATES])
        sli_up_ward = tf.concat([sli_up_ward, a_sli_up_ward], 0)

        a_sli_sp_p = tf.slice(ph_sp_p, [t.t.pos(node.name)], [1])
        sli_sp_p= tf.concat([sli_sp_p, a_sli_sp_p], 0)

        a_sli_A = tf.slice(ph_A, [0,0, t.t.pos(node.name)], [N_HIDDEN_STATES, N_HIDDEN_STATES, 1 ])
        a_sli_A = tf.transpose(a_sli_A, perm=[2,1, 0])
        sli_A= tf.concat([sli_A, a_sli_A], 0)


        #den
        a_sli_in_prior = tf.slice(var_in_prior, [0, t.t.pa(node.name).name], [N_HIDDEN_STATES,1])
        sli_in_prior = tf.concat([sli_in_prior,a_sli_in_prior],1)

        a_sli_a_up_ward = tf.slice(var_a_up_ward, [t.t.pa(node.name).name, 0, 0 ], [1,N_HIDDEN_STATES,MAX_CHILD])
        sli_var_a_up_ward = tf.concat([sli_var_a_up_ward,a_sli_a_up_ward],0)



    # per il numeratore

    sli_E = tf.expand_dims(sli_E, 0)
    sli_E = tf.tile(sli_E, [N_HIDDEN_STATES, 1, 1])
    sli_E = tf.transpose(sli_E, perm=[1,0,2])

    sli_up_ward = tf.expand_dims(sli_up_ward, 0)
    sli_up_ward = tf.tile(sli_up_ward, [N_HIDDEN_STATES, 1, 1])
    sli_up_ward = tf.transpose(sli_up_ward, perm=[1, 0, 2])

    sli_sp_p = tf.expand_dims(sli_sp_p, 0)
    sli_sp_p = tf.expand_dims(sli_sp_p, 0)
    sli_sp_p = tf.tile(sli_sp_p, [N_HIDDEN_STATES,N_HIDDEN_STATES, 1])
    sli_sp_p = tf.transpose(sli_sp_p, perm=[2, 1, 0])

    numerator = tf.multiply(sli_E, sli_up_ward)
    numerator = tf.multiply(numerator, sli_sp_p)
    numerator = tf.multiply(numerator, sli_A)


    # per il denominatore
    a_sp_p = tf.expand_dims(ph_sp_p, 0)
    a_sp_p = tf.expand_dims(a_sp_p, 0)
    a_sp_p = tf.tile(a_sp_p, [len( t.struct[i]),N_HIDDEN_STATES, 1])

    to_sum = tf.multiply(a_sp_p,sli_var_a_up_ward)
    added = tf.reduce_sum(to_sum, [2])  # sommo nella dim 2

    sli_in_prior = tf.transpose(sli_in_prior, perm=[1, 0])

    denominator = tf.multiply(sli_in_prior,added)
    denominator = tf.expand_dims(denominator, 1)
    denominator = tf.tile(denominator, [1, N_HIDDEN_STATES, 1])

    ris_24 = tf.multiply(numerator,denominator)

    head = tf.slice(var_EE, [0, 0, 0], [t.struct[i][0].name, N_HIDDEN_STATES,N_HIDDEN_STATES])

    if (t.struct[i][-1].name +1 )!= t.size:
        tail = tf.slice(var_EE, [t.struct[i + 1][0].name, 0, 0], [t.size - t.struct[i + 1][0].name, N_HIDDEN_STATES,N_HIDDEN_STATES])
        var_EE = tf.concat([head, ris_24, tail], 0)
    else:
        var_EE = tf.concat([head, ris_24], 0)

    ris_25 = tf.reduce_sum(ris_24, [1])  # !!!!!!!!!!!!!!!!!!!!!!!!!!!!! qui non sono sicuro se sto sommando le j o le i

    head = tf.slice(var_E, [0, 0], [t.struct[i][0].name, N_HIDDEN_STATES])

    if (t.struct[i][-1].name + 1) != t.size:
        tail = tf.slice(var_E, [t.struct[i + 1][0].name, 0],
                        [t.size - t.struct[i + 1][0].name, N_HIDDEN_STATES])
        var_E = tf.concat([head, ris_25, tail], 0)
    else:
        var_E = tf.concat([head, ris_25], 0)
#------------------------

with tf.Session() as sess:

    """writer = tf.summary.FileWriter("/home/simone/Documents/università/TESI/codici/reverse_up_down")
    writer.add_graph(sess.graph)"""

    sess.run(tf.global_variables_initializer(),)
    print(sess.run([var_E,var_EE], {ph_bi: bi, ph_pi: pi,ph_sp_p: sp_p, ph_A: A,ph_in_prior: in_prior, ph_A: A}))








