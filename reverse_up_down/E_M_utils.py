import numpy as np
import tensorflow as tf
from tre_simple import *



class modello(object):
    def __init__(self,pi=None,sp_p=None,A=None,bi=None):

        #da riempire in modo senzato.... probabilità ecc
        if self.pi is None:
            None
        if self.sp_p is None:
            None
        if self.A is None:
            None
        if self.bi is None:
            None

    def training(self,epche,lista_alberi):
        None

    def test(self,lista_alberi):
        None



def modello(epoche,dataset):
    #dataset :lista di alberi
    # pi, e, a , fi ecc relative al modello
    # lista di var_EE da aggiornare
    i=0;
    n=0;
    for i in range(i,epoche):

        for j in range(i,n):
            None
            # con i parametri da aggiustare
            # E-STEP
           # (var_EE) = Reversed_Upward_Downward(var_E, var_EE, ph_sp_p, ph_A, ph_bi, ph_pi, var_in_prior, var_a_up_ward,
           #                                     var_up_ward, N_HIDDEN_STATES, MAX_CHILD, t)


        #M-STEP
        # aggiusto i parametri con lista di var_EE



#||||||||||||||||||||||||||||||||||||||||||||||||||||||||E-STEP||||||||||||||||||||||||||||||||||||||||||||||||||||||||

def Reversed_Upward_Downward(var_E, var_EE, ph_sp_p, ph_A, ph_bi, ph_pi, var_in_prior, var_a_up_ward, var_up_ward, N_HIDDEN_STATES, MAX_CHILD, t):

    var_up_ward = compute_17(ph_bi,ph_pi,var_up_ward,N_HIDDEN_STATES,t)

    var_in_prior = compute_internal_node_prior(var_in_prior, ph_sp_p, ph_A, t, N_HIDDEN_STATES)

    #up step
    for i in range(len(t.struct) - 2, -1, -1):
        var_a_up_ward = compute_21(ph_A,var_in_prior,var_a_up_ward,var_up_ward,i,t,N_HIDDEN_STATES,MAX_CHILD)
        var_up_ward = compute_19(ph_bi, ph_sp_p, var_a_up_ward, var_in_prior, var_up_ward, t,i, N_HIDDEN_STATES, MAX_CHILD)

    var_E = set_base_case(var_up_ward,var_E,t,N_HIDDEN_STATES)

    # down step
    for i in range(1, len(t.struct) ):
        ris_24 = compute_24(ph_sp_p, ph_A,var_E, var_EE, var_up_ward, var_in_prior, var_a_up_ward, t, i, N_HIDDEN_STATES, MAX_CHILD)
        var_EE = inglobe_ris_liv(ris_24, var_EE, t, i,  N_HIDDEN_STATES)
        var_E =  compute_25(ris_24, var_E, i, t, N_HIDDEN_STATES)

    return (var_EE)

def compute_17(ph_bi,ph_pi,var_up_ward,N_HIDDEN_STATES,t):


    aux1=tf.ones([N_HIDDEN_STATES, 0], tf.float64)
    aux2=tf.ones([N_HIDDEN_STATES, 0], tf.float64)

    for i in range(t.struct[-1][0].name,t.size):                #inefficente se il numero di nodi è elevato rispeto a
                                                                                            #  le posizioni e ai label

        sli_ph_bi = tf.slice(ph_bi, [0, t.t.get_label(i)], [N_HIDDEN_STATES, 1])
        sli_ph_pi = tf.slice(ph_pi, [0, t.t.pos(i)], [N_HIDDEN_STATES, 1])
        aux1 = tf.concat([aux1,sli_ph_bi],1)
        aux2 = tf.concat([aux2,sli_ph_pi],1)


    nume = tf.multiply(aux1,aux2)                              #Element-wise multiplication
    den = tf.einsum('ij,ji->i', tf.transpose(aux1) ,aux2)      #Einstein summation per moltiplicazione di
                                                                                # righe e colonne con lo stesso indice

    ris_17_t = tf.divide(nume,den)
    ris_17_t = tf.transpose(ris_17_t, perm=[1, 0])

    head = tf.slice(var_up_ward, [0, 0],[t.struct[-1][0].name, N_HIDDEN_STATES])
    var_up_ward = tf.concat([head, ris_17_t], 0)

    return var_up_ward

def compute_internal_node_prior(var_in_prior,ph_sp_p,ph_A,t,N_HIDDEN_STATES):

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

            children = tf.expand_dims(children, 0)  # faccio un broadcast esplicito duplicando la matrice su una nuova dimenzione  per il numero di stati nascosti
            children = tf.tile(children, [N_HIDDEN_STATES, 1, 1])
            aux_list.append(children)

            aux2 = tf.stack(aux_list, 0)  # concateno tutte le matrici N_HIDDEN_STATES*L dei node_prior per ogni gruppo
                            #  di figli di un nodo  in un unica matrice N_HIDDEN_STATES*L*(numero di nodi del livello)

        # qui moltiplicazione
        aux3 = tf.multiply(aux1, aux2)  # questa è una serie di matrici, tante quanti sono i nodi del livello esaminati

        s = tf.reduce_sum(aux3, [2, 3])  # sommo nella dimenzione 2 e 3 della matrice
        s = tf.transpose(s)

        head = tf.slice(var_in_prior, [0, 0],
                        [N_HIDDEN_STATES, t.struct[i][0].name])  # potrei farlo anche con un constant
        tail = tf.slice(var_in_prior, [0, t.struct[i][-1].name + 1],
                        [N_HIDDEN_STATES, t.size - t.struct[i][-1].name - 1])  # potrei farlo anche con un constant

        var_in_prior = tf.concat([head, s, tail], 1)  # aggiorno i nuovi valore trovati

    return var_in_prior

def compute_21(ph_A,var_in_prior,var_a_up_ward,var_up_ward,i,t,N_HIDDEN_STATES,MAX_CHILD):

    aux_up_ward = tf.ones([0, N_HIDDEN_STATES], tf.float64)
    aux_list = []
    #per ogni nodo del livello
    node_in_priors = tf.ones([N_HIDDEN_STATES, 0], tf.float64)
    for node in t.struct[i]:

        children = tf.ones([0, N_HIDDEN_STATES], tf.float64)

        # per ogni figlio del nodo del livello
        for child_node in node.children:
            child = tf.slice(var_up_ward, [child_node.name,0 ], [1, N_HIDDEN_STATES])
            children = tf.concat([children, child], 0)

        # faccio un broadcast esplicito duplicando la matrice su una nuova dimenzione  per il numero di stati nascosti
        children = tf.expand_dims(children, 0)
        children = tf.tile(children, [N_HIDDEN_STATES, 1, 1])
        children = tf.transpose(children, perm=[2,0,1])

        aux_list.append(children)
        aux_up_ward = tf.stack(aux_list, 0)

        node_in_prior = tf.slice(var_in_prior, [0, node.name], [N_HIDDEN_STATES, 1])
        node_in_priors = tf.concat([node_in_priors, node_in_prior], 1)

    # faccio un broadcast esplicito duplicando la matrice su una nuova dimenzione  per il numero di stati nascosti
    node_in_priors = tf.expand_dims(node_in_priors, 0)
    node_in_priors = tf.tile(node_in_priors, [MAX_CHILD, 1, 1])
    node_in_priors = tf.transpose(node_in_priors, perm=[2, 1, 0])

    numerator_n = tf.multiply(ph_A, aux_up_ward)
    numerator = tf.reduce_sum(numerator_n, [1])  # sommo sulla dim 1
    s = tf.divide( numerator,node_in_priors,)



    head = tf.slice(var_a_up_ward, [0, 0, 0], [t.struct[i][0].name , N_HIDDEN_STATES, MAX_CHILD])                                       # potrei farlo anche con un constant
    tail = tf.slice(var_a_up_ward, [t.struct[i][-1].name+1,0,0],
                                                        [t.size - t.struct[i][-1].name-1,N_HIDDEN_STATES, MAX_CHILD])          # potrei farlo anche con un constant

    var_a_up_ward = tf.concat([head,s,tail],0)

    return var_a_up_ward






def compute_19(ph_bi, ph_sp_p, var_a_up_ward, var_in_prior, var_up_ward, t,i, N_HIDDEN_STATES, MAX_CHILD):
    sli_ph_bi = tf.ones([N_HIDDEN_STATES, 0], tf.float64)
    second_term = tf.ones([0, N_HIDDEN_STATES, MAX_CHILD], tf.float64)
    sli_var_in_prior = tf.ones([N_HIDDEN_STATES, 0], tf.float64)

    for node in t.struct[i]:

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

    head = tf.slice(var_up_ward, [0, 0], [t.struct[i][0].name, N_HIDDEN_STATES])
    tail = tf.slice(var_up_ward, [t.struct[i + 1][0].name, 0], [t.size - t.struct[i + 1][0].name, N_HIDDEN_STATES])

    var_up_ward = tf.concat([head, ris_19, tail], 0)

    return var_up_ward

def set_base_case(var_up_ward,var_E,t,N_HIDDEN_STATES):

    base = tf.slice(var_up_ward, [0, 0], [1, N_HIDDEN_STATES])
    head = tf.slice(var_E, [0, 1], [N_HIDDEN_STATES, t.size - 1])
    head = tf.transpose(head, perm=[1, 0])
    var_E = tf.concat([base, head], 0)

    return var_E

def compute_24(ph_sp_p, ph_A,var_E, var_EE, var_up_ward, var_in_prior, var_a_up_ward, t, i, N_HIDDEN_STATES, MAX_CHILD):
    sli_E = tf.zeros([0, N_HIDDEN_STATES], tf.float64)
    sli_in_prior = tf.zeros([N_HIDDEN_STATES, 0], tf.float64)
    sli_up_ward = tf.zeros([0, N_HIDDEN_STATES], tf.float64)
    sli_sp_p = tf.zeros([0], tf.float64)
    sli_A = tf.zeros([0, N_HIDDEN_STATES, N_HIDDEN_STATES], tf.float64)
    sli_var_a_up_ward = tf.zeros([0, N_HIDDEN_STATES, MAX_CHILD], tf.float64)

    for node in t.struct[i]:
        # numer
        a_sli_E = tf.slice(var_E, [t.t.pa(node.name).name, 0], [1, N_HIDDEN_STATES])
        sli_E = tf.concat([sli_E, a_sli_E], 0)

        a_sli_up_ward = tf.slice(var_up_ward, [node.name, 0], [1, N_HIDDEN_STATES])
        sli_up_ward = tf.concat([sli_up_ward, a_sli_up_ward], 0)

        a_sli_sp_p = tf.slice(ph_sp_p, [t.t.pos(node.name)], [1])
        sli_sp_p = tf.concat([sli_sp_p, a_sli_sp_p], 0)

        a_sli_A = tf.slice(ph_A, [0, 0, t.t.pos(node.name)], [N_HIDDEN_STATES, N_HIDDEN_STATES, 1])
        a_sli_A = tf.transpose(a_sli_A, perm=[2, 1, 0])
        sli_A = tf.concat([sli_A, a_sli_A], 0)

        # den
        a_sli_in_prior = tf.slice(var_in_prior, [0, t.t.pa(node.name).name], [N_HIDDEN_STATES, 1])
        sli_in_prior = tf.concat([sli_in_prior, a_sli_in_prior], 1)

        a_sli_a_up_ward = tf.slice(var_a_up_ward, [t.t.pa(node.name).name, 0, 0], [1, N_HIDDEN_STATES, MAX_CHILD])
        sli_var_a_up_ward = tf.concat([sli_var_a_up_ward, a_sli_a_up_ward], 0)

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

    return ris_24

def inglobe_ris_liv(ris_24, var_EE, t, i,  N_HIDDEN_STATES):
    head = tf.slice(var_EE, [0, 0, 0], [t.struct[i][0].name, N_HIDDEN_STATES, N_HIDDEN_STATES])

    if (t.struct[i][-1].name + 1) != t.size:
        tail = tf.slice(var_EE, [t.struct[i + 1][0].name, 0, 0],
                        [t.size - t.struct[i + 1][0].name, N_HIDDEN_STATES, N_HIDDEN_STATES])
        var_EE = tf.concat([head, ris_24, tail], 0)
    else:
        var_EE = tf.concat([head, ris_24], 0)

    return var_EE

def compute_25(ris_24, var_E, i, t, N_HIDDEN_STATES):

    ris_25 = tf.reduce_sum(ris_24, [1])  # !!!!!!!!!!!!!!!!!!!!!!!!!!!!! qui non sono sicuro se sto sommando le j o le i

    head = tf.slice(var_E, [0, 0], [t.struct[i][0].name, N_HIDDEN_STATES])

    if (t.struct[i][-1].name + 1) != t.size:
        tail = tf.slice(var_E, [t.struct[i + 1][0].name, 0],
                        [t.size - t.struct[i + 1][0].name, N_HIDDEN_STATES])
        var_E = tf.concat([head, ris_25, tail], 0)
    else:
        var_E = tf.concat([head, ris_25], 0)

    return var_E

#||||||||||||||||||||||||||||||||||||||||||||||||||||||||M-STEP||||||||||||||||||||||||||||||||||||||||||||||||||||||||



