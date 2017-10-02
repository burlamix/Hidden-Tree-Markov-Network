import numpy as np
import tensorflow as tf
from tre_simple import *
from parser import *


#################### emf

N_HIDDEN_STATES  = 4   #C

N_SYMBOLS = 366           #M


"""
#serve solo per la costruzione dell'albero lineare
MAX_LEVEL = 3
MAX_CHILD = 3          #L
t = Tree(1,1)
t.t.make_linear_tree(MAX_CHILD, MAX_LEVEL, N_SYMBOLS)
t.divide_leaves()
t.set_name()
"""
MAX_CHILD = 11          #L

t = dataset_parser()
t=t[9]
print(t)
print("----------------------------------------------------------------------------------")
#model parameters
#positional prior probability matrix --- pi
pi = np.ones((N_HIDDEN_STATES,MAX_CHILD))
#SP probability matrix --- fi
sp_p = np.ones((MAX_CHILD))
#positional state transiction probability matrix --- A
A = np.ones((N_HIDDEN_STATES,N_HIDDEN_STATES,MAX_CHILD)) # assumo che gli inidici sono j, i, l
#multinomial emision --- bi
bi = np.ones((N_HIDDEN_STATES,N_SYMBOLS))



################################################################################ tutti questi vanno dentro la funzio




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


for i in range(0,MAX_CHILD):
    sp_p[i]=i+1

w=0
for i in range(0,N_HIDDEN_STATES ):
    for j in range(0, N_HIDDEN_STATES):
        for k in range(0, MAX_CHILD):
            A[i,j,k]=w
            w=w+3


ph_pi = tf.placeholder(shape=[N_HIDDEN_STATES,MAX_CHILD], dtype=tf.float64)
ph_sp_p = tf.placeholder(shape=[MAX_CHILD], dtype=tf.float64)
ph_A = tf.placeholder(shape=[N_HIDDEN_STATES,N_HIDDEN_STATES,MAX_CHILD], dtype=tf.float64)
ph_bi = tf.placeholder(shape=[N_HIDDEN_STATES,N_SYMBOLS], dtype=tf.float64)



##################################dichiarazioni

# upward parameters beta
up_ward = np.zeros((t.size, N_HIDDEN_STATES))
a_up_ward = np.zeros((t.size, N_HIDDEN_STATES, MAX_CHILD))

# stater posterior €
post = np.ones((t.size, N_HIDDEN_STATES))
# pairwwise smoothed posterior
s_post = np.ones((MAX_CHILD, N_HIDDEN_STATES, N_HIDDEN_STATES, t.size))

# internal node prior
in_prior = np.zeros((N_HIDDEN_STATES, t.size))

# pairwise smoothed posterior
E = np.zeros((N_HIDDEN_STATES, t.size))
EE = np.zeros((t.size, N_HIDDEN_STATES, N_HIDDEN_STATES))

ph_post = tf.placeholder(shape=[t.size, N_HIDDEN_STATES], dtype=tf.float64)
ph_s_post = tf.placeholder(shape=[MAX_CHILD, N_HIDDEN_STATES, N_HIDDEN_STATES, t.size], dtype=tf.float64)
ph_in_prior = tf.placeholder(shape=[N_HIDDEN_STATES, t.size], dtype=tf.float64)


# inposto in maniera casuale il valore del node_prior solo e soltanto delle foglie
w = 1
for nodo in t.struct[-1]:
    for i in range(0, N_HIDDEN_STATES):
        in_prior[i, nodo.name] = w
        w = w + 1

'''z = 1
for i in range(0, t.size):
    for j in range(0, N_HIDDEN_STATES):
        up_ward[i, j] = z
        z = z + 1
w = 0
for i in range(0, t.size):
    for j in range(0, N_HIDDEN_STATES):
        for k in range(0, MAX_CHILD):
            a_up_ward[i, j, k] = w
            w = w + 1

'''




z = 0
for j in range(0, t.size):
    for i in range(0, N_HIDDEN_STATES):
        E[i, j] = 0
        z = z + 1

init_prior = tf.constant(in_prior, dtype=tf.float64)
var_in_prior = tf.get_variable('var_in_prior', initializer=init_prior )

init_a_up_ward = tf.constant(a_up_ward, dtype=tf.float64)
var_a_up_ward = tf.get_variable('var_a_up_ward', initializer=init_a_up_ward)
init_up_ward = tf.constant(up_ward, dtype=tf.float64)
var_up_ward = tf.get_variable('var_up_ward', initializer=init_up_ward)

init_E = tf.constant(E, dtype=tf.float64)
var_E = tf.get_variable('E', initializer=init_E)
init_EE = tf.constant(EE, dtype=tf.float64)
var_EE = tf.get_variable('EE', initializer=init_EE)



#----------------------------17------------------------------
    aux1 = tf.ones([N_HIDDEN_STATES, 0], tf.float64)
    aux2 = tf.ones([N_HIDDEN_STATES, 0], tf.float64)


    #taglio e concateno i vari valori relativi ad ogni B(i) così da eseguire allo stesso momento la moltiplicazione in parallelo su GPU
    for nodi in t.struct[-1]:  # inefficente se il numero di nodi è elevato rispeto a
        #  le posizioni e ai label
        sli_ph_bi = tf.slice(ph_bi, [0, t.t.get_label(nodi.name)], [N_HIDDEN_STATES, 1])
        sli_ph_pi = tf.slice(ph_pi, [0, t.t.pos(nodi.name)], [N_HIDDEN_STATES, 1])
        aux1 = tf.concat([aux1, sli_ph_bi], 1)
        aux2 = tf.concat([aux2, sli_ph_pi], 1)

    nume = tf.multiply(aux1, aux2)  # Element-wise multiplication
    den = tf.einsum('ij,ji->i', tf.transpose(aux1), aux2)  # Einstein summation per moltiplicazione di
    # righe e colonne con lo stesso indice

    ris_17_t = tf.divide(nume, den)
    ris_17_t = tf.transpose(ris_17_t, perm=[1, 0])

    #taglio la parte non foglie di var_up_ward così da concatenarlo dopo
    head = tf.slice(var_up_ward, [0, 0], [t.size - len(t.struct[-1]), N_HIDDEN_STATES])
    var_up_ward = tf.concat([head, ris_17_t], 0)

#----------------------------compute_internal_node_prior (20) ------------------------------

aux1 = tf.multiply(ph_sp_p, ph_A)  # broadcast implicito

# per ogni livello dell'albero

for i in range(len(t.struct) - 2, -1, -1):

    aux2 = tf.ones([0, N_HIDDEN_STATES], tf.float64)
    aux_list = []

    # per ogni nodo del livello
    for node in t.struct[i]:

        children = tf.ones([N_HIDDEN_STATES, 0], tf.float64)

        # per ogni figlio del nodo del livello
        for child_node in node.children:
            child = tf.slice(var_in_prior, [0, child_node.name], [N_HIDDEN_STATES, 1])  # estraggo il node prior
            children = tf.concat([children, child], 1)  # creo una matrice dei node_prior per figli di un nodo

        while (children.shape[1] < MAX_CHILD):
            pad = tf.zeros([N_HIDDEN_STATES, 1], tf.float64)
            children = tf.concat([children, pad], 1)

        children = tf.expand_dims(children,
                                  0)  # faccio un broadcast esplicito duplicando la matrice su una nuova dimenzione  per il numero di stati nascosti
        children = tf.tile(children, [N_HIDDEN_STATES, 1, 1])
        aux_list.append(children)

    aux2 = tf.stack(aux_list,
                    0)  # concateno tutte le matrici N_HIDDEN_STATES*L dei node_prior per ogni gruppo -----------------------------------------------------CONTROLLARE L'INDENTAZIONE PERCHÈ PRIMA era con uno in più e ci sta sia sbagliata
    #  di figli di un nodo  in un unica matrice N_HIDDEN_STATES*L*(numero di nodi del livello)
    # qui moltiplicazione
    aux3 = tf.multiply(aux1, aux2)  # questa è una serie di matrici, tante quanti sono i nodi del livello esaminati

    s = tf.reduce_sum(aux3, [2,
                             3])  # sommo nella dimenzione 2 e 3 della matrice________________________________________________________bisogna controllare che sia corretta i/j
    s = tf.transpose(s)

    # prelevo i valori iniziali e quelli finali che non devono essere aggiornati in questo ciclo
    head = tf.slice(var_in_prior, [0, 0],
                    # [N_HIDDEN_STATES, (t.size  -(t.size - t.struct[i][-1].name - 1) - int((s.shape[1])) )])
                    [N_HIDDEN_STATES,
                     t.struct[i][-1].name + 1 - int((s.shape[1]))])  # ricorda che questa deriva da quella sopra

    tail = tf.slice(var_in_prior, [0, t.struct[i][-1].name + 1],
                    [N_HIDDEN_STATES, t.size - t.struct[i][-1].name - 1])  # potrei farlo anche con un constant

    var_in_prior = tf.concat([head, s, tail], 1)  # aggiorno i nuovi valore trovati


#----------------------------down pass------------------------------#----------------------------down pass------------------------------#----------------------------down pass------------------------------
#for i in range(len(t.struct) - 2, -1, -1):

#  '''
for i in range(len(t.struct) - 2, -1, -1):

    print(t.struct[i])
#----------------------------21------------------------------
    #'''
    aux_up_ward = tf.ones([0, N_HIDDEN_STATES], tf.float64)
    aux_list = []
    # per ogni nodo del livello
    node_in_priors = tf.ones([N_HIDDEN_STATES, 0], tf.float64)
    for node in t.struct[i]:

        children = tf.ones([0, N_HIDDEN_STATES], tf.float64)

        # per ogni figlio del nodo del livello
        for child_node in node.children:
            child = tf.slice(var_up_ward, [child_node.name, 0], [1, N_HIDDEN_STATES])
            children = tf.concat([children, child], 0)

        while (children.shape[0] < MAX_CHILD):
            pad = tf.zeros([1, N_HIDDEN_STATES], tf.float64)
            children = tf.concat([children, pad], 0)

        # faccio un broadcast esplicito duplicando la matrice su una nuova dimenzione  per il numero di stati nascosti
        children = tf.expand_dims(children, 0)
        children = tf.tile(children, [N_HIDDEN_STATES, 1, 1])
        children = tf.transpose(children, perm=[2, 0, 1])

        aux_list.append(children)

        node_in_prior = tf.slice(var_in_prior, [0, node.name], [N_HIDDEN_STATES, 1])
        node_in_priors = tf.concat([node_in_priors, node_in_prior], 1)

    aux_up_ward = tf.stack(aux_list, 0)

    # faccio un broadcast esplicito duplicando la matrice su una nuova dimenzione  per il numero di stati nascosti
    node_in_priors = tf.expand_dims(node_in_priors, 0)
    node_in_priors = tf.tile(node_in_priors, [MAX_CHILD, 1, 1])
    node_in_priors = tf.transpose(node_in_priors, perm=[2, 1, 0])

    numerator_n = tf.multiply(ph_A, aux_up_ward)
    numerator = tf.reduce_sum(numerator_n, [
        1])  # sommo sulla dim 1________________________________________________________bisogna controllare che sia corretta i/j
    s = tf.divide(numerator, node_in_priors, )

    head = tf.slice(var_a_up_ward, [0, 0, 0], [t.struct[i][-1].name + 1 - int((s.shape[0])), N_HIDDEN_STATES,
                                               MAX_CHILD])  # potrei farlo anche con un constant

    tail = tf.slice(var_a_up_ward, [t.struct[i][-1].name + 1, 0, 0],
                    [t.size - t.struct[i][-1].name - 1, N_HIDDEN_STATES, MAX_CHILD])  # potrei farlo anche con un constant

    var_a_up_ward = tf.concat([head, s, tail], 0)


    #'''
    # ----------------------------19------------------------------    #----------------------------19------------------------------
    sli_ph_bi = tf.ones([N_HIDDEN_STATES, 0], tf.float64)
    second_term = tf.ones([0, N_HIDDEN_STATES, MAX_CHILD], tf.float64)
    sli_var_in_prior = tf.ones([N_HIDDEN_STATES, 0], tf.float64)

    for node in t.struct[i]:
        print("--->",t.struct[i])
        a_sli_ph_bi = tf.slice(ph_bi, [0, t.t.get_label(node.name)], [N_HIDDEN_STATES, 1])
        sli_ph_bi = tf.concat([sli_ph_bi, a_sli_ph_bi], 1)

        a_sli_var_a_up_ward = tf.slice(var_a_up_ward, [node.name, 0, 0], [1, N_HIDDEN_STATES, MAX_CHILD])
        second_term = tf.concat([second_term, a_sli_var_a_up_ward], 0)

        # a_sli_var_in_prior = tf.slice(var_in_prior, [0, node.name], [N_HIDDEN_STATES, 1])
        # sli_var_in_prior = tf.concat([sli_var_in_prior, a_sli_var_in_prior], 1)

    first_term = tf.expand_dims(ph_sp_p,
                                0)  # faccio un broadcast esplicito duplicando la matrice su una nuova dimenzione  per il numero di stati nascosti
    first_term = tf.expand_dims(first_term, 0)
    first_term = tf.tile(first_term, [len(t.struct[i]), N_HIDDEN_STATES, 1])

    #third_term = tf.expand_dims(sli_var_in_prior, 0)
    #third_term = tf.tile(third_term, [MAX_CHILD, 1, 1])
    #third_term = tf.transpose(third_term, perm=[2, 1, 0])

    # per in num

    #somm = tf.reduce_sum( tf.multiply( tf.multiply( first_term, second_term), third_term), [2])  # sommo sulla dim 2 (sommo le righe)
    somm = tf.reduce_sum( tf.multiply( first_term, second_term), [2])  # sommo sulla dim 2 (sommo le righe)

    sli_ph_bi = tf.transpose(sli_ph_bi, perm=[1, 0])

    numerator = tf.multiply(sli_ph_bi, somm)

    # per il den
    bb = tf.expand_dims(sli_ph_bi, 2)
    bb = tf.tile(bb, [1, 1, MAX_CHILD])

    #denominator_int = tf.reduce_sum(
    #tf.multiply(tf.multiply(tf.multiply(bb, first_term), second_term), third_term), [2, 1])  # sommo sulla dim 2 (sommo le righe)

    denominator_int = tf.reduce_sum(tf.multiply(tf.multiply(bb, first_term), second_term), [2, 1])  # sommo sulla dim 2 (sommo le righe)

    denominator = tf.expand_dims(denominator_int, 1)
    denominator = tf.tile(denominator, [1, N_HIDDEN_STATES])  # questo è il Nu uno per ogni nodo, va duplicato per ogni stato nascosto di un nodo
    # finale

    ris_19 = tf.divide(numerator, denominator)

    print("prima",var_up_ward)
   # head = tf.slice(var_up_ward, [0, 0],  [t.size - t.struct[i][-1].name-1,N_HIDDEN_STATES])
   # tail = tf.slice(var_up_ward, [int(head.shape[0]) +int(ris_19.shape[0]), 0], [t.struct[i][-1].name  - int((ris_19.shape[0])), N_HIDDEN_STATES])


    head = tf.slice(var_up_ward, [0, 0], [t.struct[i][-1].name + 1 - int((ris_19.shape[0])), N_HIDDEN_STATES])
    tail = tf.slice(var_up_ward, [t.struct[i][-1].name+1,0],  [t.size - t.struct[i][-1].name-1,N_HIDDEN_STATES])

    var_up_ward = tf.concat([head, ris_19, tail], 0)

#'''




#----------------------------17------------------------------#----------------------------17------------------------------




with tf.Session() as sess:

    """writer = tf.summary.FileWriter("/home/simone/Documents/università/TESI/codici/reverse_up_down")
    writer.add_graph(sess.graph)"""

    sess.run(tf.global_variables_initializer(),)
    print(sess.run([var_a_up_ward,var_up_ward], {ph_bi: bi, ph_pi: pi,ph_sp_p: sp_p, ph_A: A}))

