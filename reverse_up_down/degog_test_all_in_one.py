import numpy as np
import tensorflow as tf
from datetime import datetime
from E_M_utils import *
from parser import *

N_HIDDEN_STATES  = 3   #C

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
MAX_CHILD = 33          #L

t = dataset_parser()
t = t[13]
print(t)

#model parameters
#nel caso non vengano passati dei valori iniziali ai parametri essi venono inizializati random
ph_pi = random_sum_one2(1, N_HIDDEN_STATES, MAX_CHILD)
ph_sp_p = random_sum_one1(MAX_CHILD)
ph_A = random_sum_one3(1, N_HIDDEN_STATES, N_HIDDEN_STATES, MAX_CHILD)
ph_bi = random_sum_one2(0, N_HIDDEN_STATES, N_SYMBOLS)



#######################################################àà#####################################################################start debug


# upward parameters beta
up_ward = np.zeros((t.size, N_HIDDEN_STATES))
a_up_ward = np.ones((t.size, N_HIDDEN_STATES, MAX_CHILD))

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

w = 1
for i in range(0, N_HIDDEN_STATES):
    for j in range(t.size - len(t.struct[-1]), t.size):
        in_prior[i, j] = w
        w = w + 1

print(in_prior)
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


##############################17#############################################################################################

aux1 = tf.ones([N_HIDDEN_STATES, 0], tf.float64)
aux2 = tf.ones([N_HIDDEN_STATES, 0], tf.float64)

label = []
posizione = []
for node in t.struct[-1]:
    label.append(node.label)
    posizione.append(node.father.children.index(node))

    aaux1 = tf.gather(ph_bi, label,axis=1)
    aaux2 = tf.gather(ph_pi, posizione,axis=1)



for i in range(t.struct[-1][0].name, t.size):  # inefficente se il numero di nodi è elevato rispeto a
    #  le posizioni e ai label

    sli_ph_bi = tf.slice(ph_bi, [0, t.t.get_label(i)], [N_HIDDEN_STATES, 1])
    sli_ph_pi = tf.slice(ph_pi, [0, t.t.pos(i)], [N_HIDDEN_STATES, 1])
    aux1 = tf.concat([aux1, sli_ph_bi], 1)
    aux2 = tf.concat([aux2, sli_ph_pi], 1)




nume = tf.multiply(aux1, aux2)  # Element-wise multiplication
den = tf.einsum('ij,ji->i', tf.transpose(aux1), aux2)  # Einstein summation per moltiplicazione di
# righe e colonne con lo stesso indice

ris_17_t = tf.divide(nume, den)
ris_17_t = tf.transpose(ris_17_t, perm=[1, 0])

head = tf.slice(var_up_ward, [0, 0], [t.struct[-1][0].name, N_HIDDEN_STATES])
var_up_ward = tf.concat([head, ris_17_t], 0)

##############################compute_internal_node_prior#############################################################################################

starta = datetime.now()

aux1 = tf.multiply(ph_sp_p, ph_A)  # broadcast implicito

# per ogni livello dell'albero

for i in range(len(t.struct) - 2, -1, -1):

    nomi_figli = []
    for node in t.struct[i]:

        nomi_figli.append([])
        k=0
        for child_node in node.children:
            k=k+1
            nomi_figli[-1].append(child_node.name)
        for j in range (k,MAX_CHILD):
            nomi_figli[-1].append(0)

    #print(nomi_figli)
    aux2 = tf.gather(var_in_prior, nomi_figli,axis=1)
    aux2 = tf.transpose(aux2, perm=[1, 0, 2])
    aux2 = tf.expand_dims(aux2, 1)
    aux2 = tf.tile(aux2, [1, N_HIDDEN_STATES, 1, 1])


    #  di figli di un nodo  in un unica matrice N_HIDDEN_STATES*L*(numero di nodi del livello)
    # qui moltiplicazione
    aux3 = tf.multiply(aux1, aux2)  # questa è una serie di matrici, tante quanti sono i nodi del livello esaminati

    s = tf.reduce_sum(aux3, [2,3])  # sommo nella dimenzione 2 e 3 della matrice________________________________________________________bisogna controllare che sia corretta i/j
    s = tf.transpose(s)

    # prelevo i valori iniziali e quelli finali che non devono essere aggiornati in questo ciclo
    head = tf.slice(var_in_prior, [0, 0],
                    # [N_HIDDEN_STATES, (t.size  -(t.size - t.struct[i][-1].name - 1) - int((s.shape[1])) )])
                    [N_HIDDEN_STATES,
                     t.struct[i][-1].name + 1 - int((s.shape[1]))])  # ricorda che questa deriva da quella sopra

    tail = tf.slice(var_in_prior, [0, t.struct[i][-1].name + 1],
                    [N_HIDDEN_STATES, t.size - t.struct[i][-1].name - 1])  # potrei farlo anche con un constant

    var_in_prior = tf.concat([head, s, tail], 1)  # aggiorno i nuovi valore trovati
print("tot   :  ", (datetime.now() - starta).total_seconds() * 1000)


    ##############################up step                #############################################################################################

    ##############################     21                #############################################################################################

    #up step
for i in range(len(t.struct) - 2, -1, -1):

    nomi_figli = []
    nomi_nodi = []
    for node in t.struct[i]:
        nomi_nodi.append(node.name)

        nomi_figli.append([])
        k = 0
        for child_node in node.children:
            k = k + 1
            nomi_figli[-1].append(child_node.name)
        for j in range(k, MAX_CHILD):
            nomi_figli[-1].append(0)


    aux_up_ward = tf.gather(var_up_ward, nomi_figli)
    aux_up_ward = tf.expand_dims(aux_up_ward, 1)
    aux_up_ward = tf.tile(aux_up_ward, [1, N_HIDDEN_STATES, 1, 1])
    aux_up_ward = tf.transpose(aux_up_ward, perm=[0, 3, 1, 2])


    node_in_priors = tf.gather(var_in_prior, nomi_nodi, axis=1)  # questo easy come gli altri di prima§§§§§§§§§§§§§§§§§§§


    # faccio un broadcast esplicito duplicando la matrice su una nuova dimenzione  per il numero di stati nascosti
    node_in_priors = tf.expand_dims(node_in_priors, 0)
    node_in_priors = tf.tile(node_in_priors, [MAX_CHILD, 1, 1])
    node_in_priors = tf.transpose(node_in_priors, perm=[2, 1, 0])


    numerator_n = tf.multiply(ph_A, aux_up_ward)
    numerator = tf.reduce_sum(numerator_n, [1])  # sommo sulla dim 1________________________________________________________bisogna controllare che sia corretta i/j
    s = tf.divide(numerator, node_in_priors )

    head = tf.slice(var_a_up_ward, [0, 0, 0], [t.struct[i][-1].name + 1 - int((s.shape[0])), N_HIDDEN_STATES,
                                               MAX_CHILD])  # potrei farlo anche con un constant

    tail = tf.slice(var_a_up_ward, [t.struct[i][-1].name + 1, 0, 0],
                    [t.size - t.struct[i][-1].name - 1, N_HIDDEN_STATES,
                     MAX_CHILD])  # potrei farlo anche con un constant
    print(var_a_up_ward)
    var_a_up_ward = tf.concat([head, s, tail], 0)

    '''
    ##############################          19                #############################################################################################

    sli_var_in_prior = tf.ones([N_HIDDEN_STATES, 0], tf.float64)

    labels = []
    nomi = []
    for node in t.struct[i]:
        labels.append(node.label)
        nomi.append(node.name)

    sli_ph_bi = tf.gather(ph_bi, labels, axis=1)
    second_term = tf.gather(var_a_up_ward, nomi)



        #a_sli_var_in_prior = tf.slice(var_in_prior, [0, node.name], [N_HIDDEN_STATES, 1])  # questi sono commentati per quello che mi ha detto il bacciu!
        #sli_var_in_prior = tf.concat([sli_var_in_prior, a_sli_var_in_prior], 1)

    first_term = tf.expand_dims(ph_sp_p,
                                0)  # faccio un broadcast esplicito duplicando la matrice su una nuova dimenzione  per il numero di stati nascosti
    first_term = tf.expand_dims(first_term, 0)
    first_term = tf.tile(first_term, [len(t.struct[i]), N_HIDDEN_STATES, 1])

    # third_term = tf.expand_dims(sli_var_in_prior, 0)
    # third_term = tf.tile(third_term, [MAX_CHILD, 1, 1])
    # third_term = tf.transpose(third_term, perm=[2, 1, 0])

    # per in num

    # somm = tf.reduce_sum( tf.multiply( tf.multiply( first_term, second_term), third_term), [2])  # sommo sulla dim 2 (sommo le righe)
    somm = tf.reduce_sum(tf.multiply(first_term, second_term), [2])  # sommo sulla dim 2 (sommo le righe)

    sli_ph_bi = tf.transpose(sli_ph_bi, perm=[1, 0])

    numerator = tf.multiply(sli_ph_bi, somm)

    # per il den
    bb = tf.expand_dims(sli_ph_bi, 2)
    bb = tf.tile(bb, [1, 1, MAX_CHILD])

    # denominator_int = tf.reduce_sum(
    # tf.multiply(tf.multiply(tf.multiply(bb, first_term), second_term), third_term), [2, 1])  # sommo sulla dim 2 (sommo le righe)

    denominator_int = tf.reduce_sum(tf.multiply(tf.multiply(bb, first_term), second_term),
                                    [2, 1])  # sommo sulla dim 2 (sommo le righe)

    denominator = tf.expand_dims(denominator_int, 1)
    denominator = tf.tile(denominator, [1,
                                        N_HIDDEN_STATES])  # questo è il Nu uno per ogni nodo, va duplicato per ogni stato nascosto di un nodo
    # finale

    ris_19 = tf.divide(numerator, denominator)

    head = tf.slice(var_up_ward, [0, 0], [t.struct[i][-1].name + 1 - int((ris_19.shape[0])), N_HIDDEN_STATES])
    tail = tf.slice(var_up_ward, [t.struct[i][-1].name + 1, 0],
                    [t.size - t.struct[i][-1].name - 1, N_HIDDEN_STATES])

    var_up_ward = tf.concat([head, ris_19, tail], 0)
##############################          set_base_case                #############################################################################################
base = tf.slice(var_up_ward, [0, 0], [1, N_HIDDEN_STATES])
head = tf.slice(var_E, [0, 1], [N_HIDDEN_STATES, t.size - 1])
head = tf.transpose(head, perm=[1, 0])
var_E = tf.concat([base, head], 0)

##############################          down step                #############################################################################################
# down step
for i in range(1, len(t.struct)  ):

    ##############################           compute_24                #############################################################################################
    sli_E = tf.zeros([0, N_HIDDEN_STATES], tf.float64)
    sli_in_prior = tf.zeros([N_HIDDEN_STATES, 0], tf.float64)
    sli_up_ward = tf.zeros([0, N_HIDDEN_STATES], tf.float64)
    sli_sp_p = tf.zeros([0], tf.float64)
    sli_A = tf.zeros([0, N_HIDDEN_STATES, N_HIDDEN_STATES], tf.float64)
    sli_var_a_up_ward = tf.zeros([0, N_HIDDEN_STATES, MAX_CHILD], tf.float64)


    padri=[]
    posizione=[]
    nomi_nodi=[]
    for node in t.struct[i]:
        nomi_nodi.append(node.name)
        padri.append(node.father.name)
        posizione.append(node.father.children.index(node))


    sli_E = tf.gather(var_E, padri)
    sli_up_ward = tf.gather(var_up_ward, nomi_nodi)
    sli_sp_p = tf.gather(ph_sp_p, posizione)
    sli_A = tf.gather(ph_A, posizione, axis=2)
    sli_A = tf.transpose(sli_A, perm=[2, 1, 0])
    sli_in_prior = tf.gather(var_in_prior, padri, axis=1)
    sli_var_a_up_ward = tf.gather(var_a_up_ward, padri)




    # per il numeratore

    sli_E = tf.expand_dims(sli_E, 0)
    sli_E = tf.tile(sli_E, [N_HIDDEN_STATES, 1, 1])
    sli_E = tf.transpose(sli_E, perm=[1, 0, 2])

    sli_up_ward = tf.expand_dims(sli_up_ward, 0)
    sli_up_ward = tf.tile(sli_up_ward, [N_HIDDEN_STATES, 1, 1])
    sli_up_ward = tf.transpose(sli_up_ward, perm=[1, 0, 2])

    sli_sp_p = tf.expand_dims(sli_sp_p, 0)
    sli_sp_p = tf.expand_dims(sli_sp_p, 0)
    sli_sp_p = tf.tile(sli_sp_p, [N_HIDDEN_STATES, N_HIDDEN_STATES, 1])
    sli_sp_p = tf.transpose(sli_sp_p, perm=[2, 1, 0])

    numerator = tf.multiply(sli_E, sli_up_ward)
    numerator = tf.multiply(numerator, sli_sp_p)
    numerator = tf.multiply(numerator, sli_A)

    # per il denominatore
    a_sp_p = tf.expand_dims(ph_sp_p, 0)
    a_sp_p = tf.expand_dims(a_sp_p, 0)
    a_sp_p = tf.tile(a_sp_p, [len(t.struct[i]), N_HIDDEN_STATES, 1])

    to_sum = tf.multiply(a_sp_p, sli_var_a_up_ward)
    added = tf.reduce_sum(to_sum, [2])  # sommo nella dim 2

    sli_in_prior = tf.transpose(sli_in_prior, perm=[1, 0])

    denominator = tf.multiply(sli_in_prior, added)
    denominator = tf.expand_dims(denominator, 1)
    denominator = tf.tile(denominator, [1, N_HIDDEN_STATES, 1])

    ris_24 = tf.multiply(numerator, denominator)

    ##############################          inglobe_ris_liv               #############################################################################################
head = tf.slice(var_EE, [0, 0, 0], [t.struct[i][-1].name + 1 - int((ris_24.shape[0])), N_HIDDEN_STATES, N_HIDDEN_STATES])
tail = tf.slice(var_EE, [t.struct[i][-1].name + 1, 0, 0], [t.size - t.struct[i][-1].name - 1, N_HIDDEN_STATES, N_HIDDEN_STATES])



var_EE = tf.concat([head, ris_24, tail], 0)
##############################          compute_25               #############################################################################################
ris_25 = tf.reduce_sum(ris_24,
                       [1])  # !!!!!!!!!!!!!!!!!!!!!!!!!!!!! qui non sono sicuro se sto sommando le j o le i

head = tf.slice(var_E, [0, 0], [t.struct[i][-1].name + 1 - int((ris_24.shape[0])),
                                N_HIDDEN_STATES])  # _________________-da controllare la dim giusta in shape
tail = tf.slice(var_E, [t.struct[i][-1].name + 1, 0], [t.size - t.struct[i][-1].name - 1, N_HIDDEN_STATES])

var_E = tf.concat([head, ris_25, tail], 0)

'''


with tf.Session() as sess:

    """writer = tf.summary.FileWriter("/home/simone/Documents/università/TESI/codici/reverse_up_down")
    writer.add_graph(sess.graph)"""

    sess.run(tf.global_variables_initializer(),)
    print(sess.run([var_a_up_ward]))



