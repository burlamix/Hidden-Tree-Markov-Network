import numpy as np
import tensorflow as tf
from tre_simple import *
from parser import *
from datetime import datetime

N_HIDDEN_STATES = 3
N_SYMBOLS = 367
MAX_CHILD = 32


def training(data_set,epoche,pi=None,sp_p=None,A=None,bi=None):

    n=0

    #N_HIDDEN_STATES da 2 a 20 non di più, va calcolato l'algoritmo per i vari valori, che fanno cambiare il tutto di molto IMPORTANTE


    scope_tree = "scope_n0"

    #nel caso non vengano passati dei valori iniziali ai parametri essi venono inizializati random
    if pi is None:
        pi = random_sum_one2(1, N_HIDDEN_STATES, MAX_CHILD)
    if sp_p is None:
        sp_p = random_sum_one1(MAX_CHILD)
    if A is None:
        A = random_sum_one3(1, N_HIDDEN_STATES, N_HIDDEN_STATES, MAX_CHILD)
    if bi is None:
        bi = random_sum_one2(0, N_HIDDEN_STATES, N_SYMBOLS)

    #per il numero delle epoco eseguo l'E-M

    for i in range(0, epoche):
        print("EPOCA: ",i)
        #eseguo E-STEP per ogni albero nel dataset
        var_EE_list = []
        var_E_list = []
        for j in range(0,len(data_set)):

            scope_tree=scope_tree[:-len(str(j-1))]+str(j)
            with tf.variable_scope(scope_tree):
                (var_EE, var_E) = Reversed_Upward_Downward(sp_p, A, bi, pi, N_HIDDEN_STATES, MAX_CHILD, data_set[j])
                print("         E-step")
                var_EE_list.append(var_EE)
                var_E_list.append(var_E)

        new_pi, new_sp_p, new_A, new_bi = M_step(var_EE_list,var_E_list,data_set)
        print("         M-step")

        pi = new_pi
        sp_p = new_sp_p
        A = new_A
        bi = new_bi

        t_pi = random_sum_one2(1, N_HIDDEN_STATES, MAX_CHILD)
        t_sp_p = random_sum_one1(MAX_CHILD)
        t_A = random_sum_one3(1, N_HIDDEN_STATES, N_HIDDEN_STATES, MAX_CHILD)
        t_bi = random_sum_one2(0, N_HIDDEN_STATES, N_SYMBOLS)

    return pi,sp_p,A,bi,t_pi,t_sp_p,t_A,t_bi


#||||||||||||||||||||||||||||||||||||||||||||||||||||||||E-STEP||||||||||||||||||||||||||||||||||||||||||||||||||||||||

def Reversed_Upward_Downward(ph_sp_p, ph_A, ph_bi, ph_pi, N_HIDDEN_STATES, MAX_CHILD, t):


    # upward parameters beta
    up_ward = np.ones((t.size, N_HIDDEN_STATES))
    a_up_ward = np.ones((t.size, N_HIDDEN_STATES, MAX_CHILD))

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

    '''
        z = 1
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

        z = 0
        for j in range(0, t.size):
            for i in range(0, N_HIDDEN_STATES):
                E[i, j] = 0
                z = z + 1
    '''

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


###########################################################################################################################

    var_up_ward = compute_17(ph_bi,ph_pi,var_up_ward,N_HIDDEN_STATES,t)

    var_in_prior = compute_internal_node_prior(var_in_prior, ph_sp_p, ph_A, t, N_HIDDEN_STATES,MAX_CHILD)

    #up step
    for i in range(len(t.struct) - 2, -1, -1):

        var_a_up_ward = compute_21(ph_A,var_in_prior,var_a_up_ward,var_up_ward,i,t,N_HIDDEN_STATES,MAX_CHILD)
        var_up_ward = compute_19(ph_bi, ph_sp_p, var_a_up_ward, var_in_prior, var_up_ward, t,i, N_HIDDEN_STATES, MAX_CHILD)

    var_E = set_base_case(var_up_ward,var_E,t,N_HIDDEN_STATES)

    # down step
    for i in range(1, len(t.struct)  ): # il -1 l'ho iserito per far compilare e non sono sicuro sia corretto___________________________________________cambialo devi fare anche le foglie______________

        ris_24 = compute_24(ph_sp_p, ph_A,var_E, var_EE, var_up_ward, var_in_prior, var_a_up_ward, t, i, N_HIDDEN_STATES, MAX_CHILD)

        var_EE = inglobe_ris_liv(ris_24, var_EE, t, i,  N_HIDDEN_STATES)

        var_E =  compute_25(ris_24, var_E, i, t, N_HIDDEN_STATES)




    return (var_EE,var_E)










def compute_17(ph_bi,ph_pi,var_up_ward,N_HIDDEN_STATES,t):

    label = []
    posizione = []
    for node in t.struct[-1]:
        label.append(node.label)
        posizione.append(node.father.children.index(node))

    aux1 = tf.gather(ph_bi, label, axis=1)
    aux2 = tf.gather(ph_pi, posizione, axis=1)

    nume = tf.multiply(aux1,aux2)                              #Element-wise multiplication
    den = tf.einsum('ij,ji->i', tf.transpose(aux1) ,aux2)      #Einstein summation per moltiplicazione di
                                                                                # righe e colonne con lo stesso indice

    ris_17_t = tf.divide(nume,den)

    ris_17_t = tf.transpose(ris_17_t, perm=[1, 0])

    head = tf.slice(var_up_ward, [0, 0],[t.struct[-1][0].name, N_HIDDEN_STATES])
    var_up_ward = tf.concat([head, ris_17_t], 0)

    return var_up_ward

def compute_internal_node_prior(var_in_prior,ph_sp_p,ph_A,t,N_HIDDEN_STATES,MAX_CHILD):

    aux1 = tf.multiply(ph_sp_p, ph_A)  # broadcast implicito

    # per ogni livello dell'albero

    for i in range(len(t.struct) - 2, -1, -1):

        nomi_figli = []
        for node in t.struct[i]:

            nomi_figli.append([])
            k = 0
            for child_node in node.children:
                k = k + 1
                nomi_figli[-1].append(child_node.name)
            for j in range(k, MAX_CHILD):
                nomi_figli[-1].append(0)

        # print(nomi_figli)
        aux2 = tf.gather(var_in_prior, nomi_figli, axis=1)
        aux2 = tf.transpose(aux2, perm=[1, 0, 2])
        aux2 = tf.expand_dims(aux2, 1)
        aux2 = tf.tile(aux2, [1, N_HIDDEN_STATES, 1, 1])

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
    return var_in_prior

def compute_21(ph_A,var_in_prior,var_a_up_ward,var_up_ward,i,t,N_HIDDEN_STATES,MAX_CHILD):

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

    node_in_priors = tf.gather(var_in_prior, nomi_nodi, axis=1)

    # faccio un broadcast esplicito duplicando la matrice su una nuova dimenzione  per il numero di stati nascosti
    node_in_priors = tf.expand_dims(node_in_priors, 0)
    node_in_priors = tf.tile(node_in_priors, [MAX_CHILD, 1, 1])
    node_in_priors = tf.transpose(node_in_priors, perm=[2, 1, 0])


    numerator_n = tf.multiply(ph_A, aux_up_ward)
    numerator = tf.reduce_sum(numerator_n, [
        1])  # sommo sulla dim 1________________________________________________________bisogna controllare che sia corretta i/j
    s = tf.divide(numerator, node_in_priors)

    head = tf.slice(var_a_up_ward, [0, 0, 0], [t.struct[i][-1].name + 1 - int((s.shape[0])), N_HIDDEN_STATES,
                                               MAX_CHILD])  # potrei farlo anche con un constant

    tail = tf.slice(var_a_up_ward, [t.struct[i][-1].name + 1, 0, 0],
                    [t.size - t.struct[i][-1].name - 1, N_HIDDEN_STATES,
                     MAX_CHILD])  # potrei farlo anche con un constant

    var_a_up_ward = tf.concat([head, s, tail], 0)

    return var_a_up_ward




def compute_19(ph_bi, ph_sp_p, var_a_up_ward, var_in_prior, var_up_ward, t,i, N_HIDDEN_STATES, MAX_CHILD):


    sli_var_in_prior = tf.ones([N_HIDDEN_STATES, 0], tf.float64)

    labels = []
    nomi = []
    for node in t.struct[i]:
        labels.append(node.label)
        nomi.append(node.name)

    sli_ph_bi = tf.gather(ph_bi, labels, axis=1)
    second_term = tf.gather(var_a_up_ward, nomi)

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


    head = tf.slice(var_up_ward, [0, 0], [t.struct[i][-1].name + 1 - int((ris_19.shape[0])), N_HIDDEN_STATES])
    tail = tf.slice(var_up_ward, [t.struct[i][-1].name+1,0],  [t.size - t.struct[i][-1].name-1,N_HIDDEN_STATES])

    var_up_ward = tf.concat([head, ris_19, tail], 0)

    return var_up_ward

def set_base_case(var_up_ward,var_E,t,N_HIDDEN_STATES):

    base = tf.slice(var_up_ward, [0, 0], [1, N_HIDDEN_STATES])
    head = tf.slice(var_E, [0, 1], [N_HIDDEN_STATES, t.size - 1])
    head = tf.transpose(head, perm=[1, 0])
    var_E = tf.concat([base, head], 0)

    return var_E

def compute_24(ph_sp_p, ph_A,var_E, var_EE, var_up_ward, var_in_prior, var_a_up_ward, t, i, N_HIDDEN_STATES, MAX_CHILD):

    padri = []
    posizione = []
    nomi_nodi = []
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

    return ris_24


#funzione che
def inglobe_ris_liv(ris_24, var_EE, t, i,  N_HIDDEN_STATES):


    head = tf.slice(var_EE, [0, 0, 0], [t.struct[i][-1].name + 1 - int((ris_24.shape[0])), N_HIDDEN_STATES,N_HIDDEN_STATES])
    tail = tf.slice(var_EE, [t.struct[i][-1].name+1,0,0 ],  [t.size - t.struct[i][-1].name-1,N_HIDDEN_STATES,N_HIDDEN_STATES])


    var_EE = tf.concat([head, ris_24, tail], 0)

    return var_EE

def compute_25(ris_24, var_E, i, t, N_HIDDEN_STATES):

    ris_25 = tf.reduce_sum(ris_24, [1])  # !!!!!!!!!!!!!!!!!!!!!!!!!!!!! qui non sono sicuro se sto sommando le j o le i

    head = tf.slice(var_E, [0, 0], [t.struct[i][-1].name + 1 - int((ris_24.shape[0])), N_HIDDEN_STATES])#_________________-da controllare la dim giusta in shape
    tail = tf.slice(var_E, [t.struct[i][-1].name+1,0 ],  [t.size - t.struct[i][-1].name-1,N_HIDDEN_STATES])

    var_E = tf.concat([head, ris_25, tail], 0)

    return var_E





#||||||||||||||||||||||||||||||||||||||||||||||||||||||||general||||||||||||||||||||||||||||||||||||||||||||||||||||||||
#funzione ausiliaria per inizializzare in modo casuale tensori di 1 dimenzione
def random_sum_one1(shape1):

    rand = tf.random_uniform([shape1], 0, 1, dtype=tf.float64)
    sum = tf.reduce_sum(rand, [0])

    rand_sum_one = tf.divide(rand, sum)

    return rand_sum_one

#funzione ausiliaria per inizializzare in modo casuale tensori di 2 dimenzioni dati l'asse di
def random_sum_one2(axe,shape1,shape2):

    rand = tf.random_uniform([shape1, shape2], 0, 1, dtype=tf.float64)
    sum = tf.reduce_sum(rand, [axe])

    #nel caso l'asse non è lo zero lo espando duplico così da poter dividere la matrice random per esso
    if axe == 1:
        sum = tf.expand_dims(sum, 1)
        sum = tf.tile(sum, [1, shape2])

    rand_sum_one = tf.divide(rand, sum)

    return rand_sum_one

#funzione ausiliaria per inizializzare in modo casuale tensori di 3 dimenzioni
def random_sum_one3(axe,shape1,shape2,shape3=None):

    rand = tf.random_uniform([shape1, shape2, shape3], 0, 1, dtype=tf.float64)
    sum = tf.reduce_sum(rand, [axe])
    sum = tf.expand_dims(sum, 1)
    sum = tf.tile(sum, [1,shape2,1])
    rand_sum_one = tf.divide(rand, sum)

    return rand_sum_one

#||||||||||||||||||||||||||||||||||||||||||||||||||||||||M-STEP||||||||||||||||||||||||||||||||||||||||||||||||||||||||


def M_step(EE_list,E_list,data_set):

    lista_prior = []
    lista_n_in = []
    aux_list_prior = []
    aux_list_sp = []
    n_l_list = np.zeros(MAX_CHILD)
    sum_N_I =0
    max_l =-1
    num_tm_yu_list = []
    den_tm_yu_list = []

    #MULTINOMIAL
    #calcolo Tm_Yu
    for i in range(0,len(EE_list)):
        tm_yu = np.zeros([int(data_set[i].size),N_SYMBOLS])
        for level in data_set[i].struct:
            for node in level:
                tm_yu[node.name,node.label]=1

        e_resized= tf.expand_dims(E_list[i],2)
        e_resized = tf.tile(e_resized, [1,1,N_SYMBOLS])

        resized_tm_yu = tf.expand_dims(tm_yu,2)             # qui posso portare fuori dal for per ottimizzare...
        resized_tm_yu = tf.tile(resized_tm_yu, [1,1,N_HIDDEN_STATES])
        resized_tm_yu = tf.transpose(resized_tm_yu,perm=[0,2,1])
        numerator_tm_yu = tf.multiply(resized_tm_yu,e_resized)
        numerator_tm_yu = tf.reduce_sum(numerator_tm_yu,[0])
        denominator_tm_yu = tf.reduce_sum(E_list[i],[0])
        denominator_tm_yu = tf.expand_dims(denominator_tm_yu,1)
        denominator_tm_yu = tf.tile(denominator_tm_yu, [1,N_SYMBOLS])
        num_tm_yu_list.append(numerator_tm_yu)
        den_tm_yu_list.append(denominator_tm_yu)

    numerator_tm_yu = tf.stack(num_tm_yu_list,0)
    denominator_tm_yu = tf.stack(den_tm_yu_list,0)

    numerator_tm_yu = tf.reduce_sum(numerator_tm_yu, [0])
    denominator_tm_yu = tf.reduce_sum(denominator_tm_yu, [0])
    result_multinomial = tf.divide(numerator_tm_yu,denominator_tm_yu)


    # calcolo il numero totale di nodi nell L-esima posizione
    # e anche il numero massimo di nodi nella l-esima posizione, così da poter dimenzionare in maniera opportuna le dim
    for i in range(0, len(EE_list)):
        lista_prior.append([])
        lista_n_in.append([])
        n_l_list=n_l_list + data_set[i].N_L
        sum_N_I =sum_N_I+data_set[i].N_I
        for l_number in data_set[i].N_L:
            if max_l < l_number:
                max_l=l_number

        for j in range(0, MAX_CHILD):
            lista_prior[i].append([])
            lista_n_in[i].append([])


    for i in range(0,len(data_set)):

        #SP & STATE TRANSICTION
        for jj in range(0, len(data_set[i].struct) - 1):
            for nodo in data_set[i].struct[jj]:
                for kk in range(0,len(nodo.children)):
                    lista_n_in[i][kk].append(nodo.children[kk].name)

        #uniformo la lunghezza così da non rendere il tensore sparso
        for in_list in lista_n_in[i]:
            start=len(in_list)
            for k in range(start,int(max_l+1)):
                in_list.append(0)

        slice = tf.gather(EE_list[i], lista_n_in[i])
        aux_list_sp.append(slice)



        #prior
        for nodo in (data_set[i].struct[-1]):
            lista_prior[i][nodo.father.children.index(nodo)].append(nodo.name)
        j=0
        for in_list in lista_prior[i]:
            start=len(in_list)
            for k in range(start,int(max_l)):
                in_list.append(0)
        slice = tf.gather(EE_list[i], lista_prior[i] )
        aux_list_prior.append(slice)

    #SP
    aux = tf.stack(aux_list_sp,0)
    summed_sp = tf.reduce_sum(aux,[4,3,2,0])#-------------------------------------------------------------CONTROLLARE L'INDICE I,J

    result_sp = tf.divide(summed_sp,sum_N_I)

    # STATE TRANSICTION (A)
    numerator_stat_tran = tf.reduce_sum(aux,[2,0])#-------------------------------------------------------------CONTROLLARE L'INDICE I,J
    denominator_stat_tran = tf.reduce_sum(aux,[4,2,0])#-------------------------------------------------------------CONTROLLARE L'INDICE I,J
    denominator_stat_tran= tf.expand_dims(denominator_stat_tran,2)
    denominator_stat_tran = tf.tile(denominator_stat_tran, [1,1,N_HIDDEN_STATES])
    result_state_trans = tf.divide(numerator_stat_tran,denominator_stat_tran)
    result_state_trans = tf.transpose(result_state_trans,[2,1,0])

    #PRIOR
    aux = tf.stack(aux_list_prior,0)
    summed_prior = tf.reduce_sum(aux,[3,2,0])#-------------------------------------------------------------CONTROLLARE L'INDICE I,J
    n_l_list= tf.expand_dims(n_l_list,1)
    n_l_list = tf.tile(n_l_list, [1,N_HIDDEN_STATES])
    result_prior = tf.divide(summed_prior,n_l_list)
    result_prior = tf.transpose(result_prior,[1,0])

    return result_prior,result_sp,result_state_trans,result_multinomial


    '''
N_HIDDEN_STATES =3
print("leggo il dataset")
data = dataset_parser()
EE_list = []
ee_list = []
E_list = []
e_list = []

for i in range(0,len(data)):
    EE = tf.placeholder(shape=[data[i].size,N_HIDDEN_STATES,N_HIDDEN_STATES], dtype=tf.float64)
    EE_list.append(EE)
    E = tf.placeholder(shape=[N_HIDDEN_STATES,data[i].size], dtype=tf.float64)
    E_list.append(E)

    ee = np.zeros([data[i].size,N_HIDDEN_STATES,N_HIDDEN_STATES])
    e = np.zeros([N_HIDDEN_STATES,data[i].size])

    w=0
    for k in range(0,N_HIDDEN_STATES):
        for j in range(0,(data[i].size)):
            e[k,j]=w
            w=w+1
    e_list.append(e)

    w=0
    for z in range(1,data[i].size):
        for j in range(0,N_HIDDEN_STATES):
            for k in range(0,N_HIDDEN_STATES):
                ee[z,j,k]=w
                w=w+1
    ee_list.append(ee)


result_prior,result_sp,result_state_trans,result_multinomial = M_step(EE_list,E_list,data)


with tf.Session() as sess:

   sess.run(tf.global_variables_initializer())
   print(sess.run([result_prior,result_sp,result_state_trans,result_multinomial], {
       EE_list[0]: ee_list[0],
       EE_list[1]: ee_list[1],
       EE_list[2]: ee_list[2],
       EE_list[3]: ee_list[3],
       EE_list[4]: ee_list[4],
       EE_list[5]: ee_list[5],
       EE_list[6]: ee_list[6],
       EE_list[7]: ee_list[7],
       EE_list[8]: ee_list[8],
       EE_list[9]: ee_list[9],
       EE_list[10]: ee_list[10],
       EE_list[11]: ee_list[11],
       EE_list[12]: ee_list[12],
       EE_list[13]: ee_list[13],
       EE_list[14]: ee_list[14],
       EE_list[15]: ee_list[15],
       EE_list[16]: ee_list[16],
       EE_list[17]: ee_list[17],
       EE_list[18]: ee_list[18],
       EE_list[19]: ee_list[19],
       EE_list[20]: ee_list[20],
       EE_list[21]: ee_list[21],
       EE_list[22]: ee_list[22],
       EE_list[23]: ee_list[23],
       EE_list[24]: ee_list[24],
       EE_list[25]: ee_list[25],
       EE_list[26]: ee_list[26],
       EE_list[27]: ee_list[27],
       EE_list[28]: ee_list[28],
       EE_list[29]: ee_list[29],
       EE_list[30]: ee_list[30],
       EE_list[31]: ee_list[31],
       EE_list[32]: ee_list[32],
       EE_list[33]: ee_list[33],
       EE_list[34]: ee_list[34],
       EE_list[35]: ee_list[35],
       EE_list[36]: ee_list[36],
       EE_list[37]: ee_list[37],
       EE_list[38]: ee_list[38],
       EE_list[39]: ee_list[39],
       EE_list[40]: ee_list[40],
       EE_list[41]: ee_list[41],
       EE_list[42]: ee_list[42],
       EE_list[43]: ee_list[43],
       EE_list[44]: ee_list[44],
       EE_list[45]: ee_list[45],
       EE_list[46]: ee_list[46],
       EE_list[47]: ee_list[47],
       EE_list[48]: ee_list[48],
       EE_list[49]: ee_list[49],
       EE_list[50]: ee_list[50],
       EE_list[51]: ee_list[51],
       EE_list[52]: ee_list[52],
       EE_list[53]: ee_list[53],
       EE_list[54]: ee_list[54],
       EE_list[55]: ee_list[55],
       EE_list[56]: ee_list[56],
       EE_list[57]: ee_list[57],
       EE_list[58]: ee_list[58],
       EE_list[59]: ee_list[59],
       EE_list[60]: ee_list[60],
       EE_list[61]: ee_list[61],
       EE_list[62]: ee_list[62],
       EE_list[63]: ee_list[63],
       EE_list[64]: ee_list[64],
       EE_list[65]: ee_list[65],
       EE_list[66]: ee_list[66],
       EE_list[67]: ee_list[67],
       EE_list[68]: ee_list[68],
       EE_list[69]: ee_list[69],
       EE_list[70]: ee_list[70],
       EE_list[71]: ee_list[71],
       EE_list[72]: ee_list[72],
       EE_list[73]: ee_list[73],
       EE_list[74]: ee_list[74],
       EE_list[75]: ee_list[75],
       EE_list[76]: ee_list[76],
       EE_list[77]: ee_list[77],
       EE_list[78]: ee_list[78],
       EE_list[79]: ee_list[79],
       EE_list[80]: ee_list[80],
       EE_list[81]: ee_list[81],
       EE_list[82]: ee_list[82],
       EE_list[83]: ee_list[83],
       EE_list[84]: ee_list[84],
       EE_list[85]: ee_list[85],
       EE_list[86]: ee_list[86],
       EE_list[87]: ee_list[87],
       EE_list[88]: ee_list[88],
       EE_list[89]: ee_list[89],
       EE_list[90]: ee_list[90],
       EE_list[91]: ee_list[91],
       EE_list[92]: ee_list[92],
       EE_list[93]: ee_list[93],
       EE_list[94]: ee_list[94],
       EE_list[95]: ee_list[95],
       EE_list[96]: ee_list[96],
       EE_list[97]: ee_list[97],
       EE_list[98]: ee_list[98],
       EE_list[99]: ee_list[99],
       EE_list[100]: ee_list[100],
       EE_list[101]: ee_list[101],
       EE_list[102]: ee_list[102],
       EE_list[103]: ee_list[103],
       EE_list[104]: ee_list[104],
       EE_list[105]: ee_list[105],
       EE_list[106]: ee_list[106],
       EE_list[107]: ee_list[107],
       EE_list[108]: ee_list[108],
       EE_list[109]: ee_list[109],
       EE_list[110]: ee_list[110],
       EE_list[111]: ee_list[111],
       EE_list[112]: ee_list[112],
       EE_list[113]: ee_list[113],
       EE_list[114]: ee_list[114],
       EE_list[115]: ee_list[115],
       EE_list[116]: ee_list[116],
       EE_list[117]: ee_list[117],
       EE_list[118]: ee_list[118],
       EE_list[119]: ee_list[119],
       EE_list[120]: ee_list[120],
       EE_list[121]: ee_list[121],
       EE_list[122]: ee_list[122],
       EE_list[123]: ee_list[123],
       EE_list[124]: ee_list[124],
       EE_list[125]: ee_list[125],
       EE_list[126]: ee_list[126],
       EE_list[127]: ee_list[127],
       EE_list[128]: ee_list[128],
       EE_list[129]: ee_list[129],
       EE_list[130]: ee_list[130],
       EE_list[131]: ee_list[131],
       EE_list[132]: ee_list[132],
       EE_list[133]: ee_list[133],
       EE_list[134]: ee_list[134],
       EE_list[135]: ee_list[135],
       EE_list[136]: ee_list[136],
       EE_list[137]: ee_list[137],
       EE_list[138]: ee_list[138],
       EE_list[139]: ee_list[139],
       EE_list[140]: ee_list[140],
       EE_list[141]: ee_list[141],
       EE_list[142]: ee_list[142],
       EE_list[143]: ee_list[143],
       EE_list[144]: ee_list[144],
       EE_list[145]: ee_list[145],
       EE_list[146]: ee_list[146],
       EE_list[147]: ee_list[147],
       EE_list[148]: ee_list[148],
       EE_list[149]: ee_list[149],
       EE_list[150]: ee_list[150],
       EE_list[151]: ee_list[151],
       EE_list[152]: ee_list[152],
       EE_list[153]: ee_list[153],
       EE_list[154]: ee_list[154],
       EE_list[155]: ee_list[155],
       EE_list[156]: ee_list[156],
       EE_list[157]: ee_list[157],
       EE_list[158]: ee_list[158],
       EE_list[159]: ee_list[159],
       EE_list[160]: ee_list[160],
       EE_list[161]: ee_list[161],
       EE_list[162]: ee_list[162],
       EE_list[163]: ee_list[163],
       EE_list[164]: ee_list[164],
       EE_list[165]: ee_list[165],
       EE_list[166]: ee_list[166],
       EE_list[167]: ee_list[167],
       EE_list[168]: ee_list[168],
       EE_list[169]: ee_list[169],
       EE_list[170]: ee_list[170],
       EE_list[171]: ee_list[171],
       EE_list[172]: ee_list[172],
       EE_list[173]: ee_list[173],
       EE_list[174]: ee_list[174],
       EE_list[175]: ee_list[175],
       EE_list[176]: ee_list[176],
       EE_list[177]: ee_list[177],
       EE_list[178]: ee_list[178],
       EE_list[179]: ee_list[179],
       EE_list[180]: ee_list[180],
       EE_list[181]: ee_list[181],
       EE_list[182]: ee_list[182],
       EE_list[183]: ee_list[183],
       EE_list[184]: ee_list[184],
       EE_list[185]: ee_list[185],
       EE_list[186]: ee_list[186],
       EE_list[187]: ee_list[187],
       EE_list[188]: ee_list[188],
       EE_list[189]: ee_list[189],
       EE_list[190]: ee_list[190],
       EE_list[191]: ee_list[191],
       EE_list[192]: ee_list[192],
       EE_list[193]: ee_list[193],
       EE_list[194]: ee_list[194],
       EE_list[195]: ee_list[195],
       EE_list[196]: ee_list[196],
       EE_list[197]: ee_list[197],
       EE_list[198]: ee_list[198],
       EE_list[199]: ee_list[199],
       EE_list[200]: ee_list[200],
       EE_list[201]: ee_list[201],
       EE_list[202]: ee_list[202],
       EE_list[203]: ee_list[203],
       EE_list[204]: ee_list[204],
       EE_list[205]: ee_list[205],
       EE_list[206]: ee_list[206],
       EE_list[207]: ee_list[207],
       EE_list[208]: ee_list[208],
       EE_list[209]: ee_list[209],
       EE_list[210]: ee_list[210],
       EE_list[211]: ee_list[211],
       EE_list[212]: ee_list[212],
       EE_list[213]: ee_list[213],
       EE_list[214]: ee_list[214],
       EE_list[215]: ee_list[215],
       EE_list[216]: ee_list[216],
       EE_list[217]: ee_list[217],
       EE_list[218]: ee_list[218],
       EE_list[219]: ee_list[219],
       EE_list[220]: ee_list[220],
       EE_list[221]: ee_list[221],
       EE_list[222]: ee_list[222],
       EE_list[223]: ee_list[223],
       EE_list[224]: ee_list[224],
       EE_list[225]: ee_list[225],
       EE_list[226]: ee_list[226],
       EE_list[227]: ee_list[227],
       EE_list[228]: ee_list[228],
       EE_list[229]: ee_list[229],
       EE_list[230]: ee_list[230],
       EE_list[231]: ee_list[231],
       EE_list[232]: ee_list[232],
       EE_list[233]: ee_list[233],
       EE_list[234]: ee_list[234],
       EE_list[235]: ee_list[235],
       EE_list[236]: ee_list[236],
       EE_list[237]: ee_list[237],
       EE_list[238]: ee_list[238],
       EE_list[239]: ee_list[239],
       EE_list[240]: ee_list[240],
       EE_list[241]: ee_list[241],
       EE_list[242]: ee_list[242],
       EE_list[243]: ee_list[243],
       EE_list[244]: ee_list[244],
       EE_list[245]: ee_list[245],
       EE_list[246]: ee_list[246],
       EE_list[247]: ee_list[247],
       EE_list[248]: ee_list[248],
       EE_list[249]: ee_list[249],
       EE_list[250]: ee_list[250],
       EE_list[251]: ee_list[251],
       EE_list[252]: ee_list[252],
       EE_list[253]: ee_list[253],
       EE_list[254]: ee_list[254],
       EE_list[255]: ee_list[255],
       EE_list[256]: ee_list[256],
       EE_list[257]: ee_list[257],
       EE_list[258]: ee_list[258],
       EE_list[259]: ee_list[259],
       EE_list[260]: ee_list[260],
       EE_list[261]: ee_list[261],
       EE_list[262]: ee_list[262],
       EE_list[263]: ee_list[263],
       EE_list[264]: ee_list[264],
       EE_list[265]: ee_list[265],
       EE_list[266]: ee_list[266],
       EE_list[267]: ee_list[267],
       EE_list[268]: ee_list[268],
       EE_list[269]: ee_list[269],
       EE_list[270]: ee_list[270],
       EE_list[271]: ee_list[271],
       EE_list[272]: ee_list[272],
       EE_list[273]: ee_list[273],
       EE_list[274]: ee_list[274],
       EE_list[275]: ee_list[275],
       EE_list[276]: ee_list[276],
       EE_list[277]: ee_list[277],
       EE_list[278]: ee_list[278],
       EE_list[279]: ee_list[279],
       EE_list[280]: ee_list[280],
       EE_list[281]: ee_list[281],
       EE_list[282]: ee_list[282],
       EE_list[283]: ee_list[283],
       EE_list[284]: ee_list[284],
       EE_list[285]: ee_list[285],
       EE_list[286]: ee_list[286],
       EE_list[287]: ee_list[287],
       EE_list[288]: ee_list[288],
       EE_list[289]: ee_list[289],
       EE_list[290]: ee_list[290],
       EE_list[291]: ee_list[291],
       EE_list[292]: ee_list[292],
       EE_list[293]: ee_list[293],
       EE_list[294]: ee_list[294],
       EE_list[295]: ee_list[295],
       EE_list[296]: ee_list[296],
       EE_list[297]: ee_list[297],
       EE_list[298]: ee_list[298],
       EE_list[299]: ee_list[299],
       EE_list[300]: ee_list[300],
       EE_list[301]: ee_list[301],
       EE_list[302]: ee_list[302],
       EE_list[303]: ee_list[303],
       EE_list[304]: ee_list[304],
       EE_list[305]: ee_list[305],
       EE_list[306]: ee_list[306],
       EE_list[307]: ee_list[307],
       EE_list[308]: ee_list[308],
       EE_list[309]: ee_list[309],
       EE_list[310]: ee_list[310],
       EE_list[311]: ee_list[311],
       EE_list[312]: ee_list[312],
       EE_list[313]: ee_list[313],
       EE_list[314]: ee_list[314],
       EE_list[315]: ee_list[315],
       EE_list[316]: ee_list[316],
       EE_list[317]: ee_list[317],
       EE_list[318]: ee_list[318],
       EE_list[319]: ee_list[319],
       EE_list[320]: ee_list[320],
       EE_list[321]: ee_list[321],
       EE_list[322]: ee_list[322],
       EE_list[323]: ee_list[323],
       EE_list[324]: ee_list[324],
       EE_list[325]: ee_list[325],
       EE_list[326]: ee_list[326],
       EE_list[327]: ee_list[327],
       EE_list[328]: ee_list[328],
       EE_list[329]: ee_list[329],
       EE_list[330]: ee_list[330],
       EE_list[331]: ee_list[331],
       EE_list[332]: ee_list[332],
       EE_list[333]: ee_list[333],
       EE_list[334]: ee_list[334],
       EE_list[335]: ee_list[335],
       EE_list[336]: ee_list[336],
       EE_list[337]: ee_list[337],
       EE_list[338]: ee_list[338],
       EE_list[339]: ee_list[339],
       EE_list[340]: ee_list[340],
       EE_list[341]: ee_list[341],
       EE_list[342]: ee_list[342],
       EE_list[343]: ee_list[343],
       EE_list[344]: ee_list[344],
       EE_list[345]: ee_list[345],
       EE_list[346]: ee_list[346],
       EE_list[347]: ee_list[347],
       EE_list[348]: ee_list[348],
       EE_list[349]: ee_list[349],
       EE_list[350]: ee_list[350],
       EE_list[351]: ee_list[351],
       EE_list[352]: ee_list[352],
       EE_list[353]: ee_list[353],
       EE_list[354]: ee_list[354],
       EE_list[355]: ee_list[355],
       EE_list[356]: ee_list[356],
       EE_list[357]: ee_list[357],
       EE_list[358]: ee_list[358],
       EE_list[359]: ee_list[359],
       EE_list[360]: ee_list[360],
       EE_list[361]: ee_list[361],
       EE_list[362]: ee_list[362],
       EE_list[363]: ee_list[363],
       EE_list[364]: ee_list[364],
       EE_list[365]: ee_list[365],
       EE_list[366]: ee_list[366],
       EE_list[367]: ee_list[367],
       EE_list[368]: ee_list[368],
       EE_list[369]: ee_list[369],
       EE_list[370]: ee_list[370],
       EE_list[371]: ee_list[371],
       EE_list[372]: ee_list[372],
       EE_list[373]: ee_list[373],
       EE_list[374]: ee_list[374],
       EE_list[375]: ee_list[375],
       EE_list[376]: ee_list[376],
       EE_list[377]: ee_list[377],
       EE_list[378]: ee_list[378],
       EE_list[379]: ee_list[379],
       EE_list[380]: ee_list[380],
       EE_list[381]: ee_list[381],
       EE_list[382]: ee_list[382],
       EE_list[383]: ee_list[383],
       EE_list[384]: ee_list[384],
       EE_list[385]: ee_list[385],
       EE_list[386]: ee_list[386],
       EE_list[387]: ee_list[387],
       EE_list[388]: ee_list[388],
       EE_list[389]: ee_list[389],
       EE_list[390]: ee_list[390],
       EE_list[391]: ee_list[391],
       EE_list[392]: ee_list[392],
       EE_list[393]: ee_list[393],
       EE_list[394]: ee_list[394],
       EE_list[395]: ee_list[395],
       EE_list[396]: ee_list[396],
       EE_list[397]: ee_list[397],
       EE_list[398]: ee_list[398],
       EE_list[399]: ee_list[399],
       EE_list[400]: ee_list[400],
       EE_list[401]: ee_list[401],
       EE_list[402]: ee_list[402],
       EE_list[403]: ee_list[403],
       EE_list[404]: ee_list[404],
       EE_list[405]: ee_list[405],
       EE_list[406]: ee_list[406],
       EE_list[407]: ee_list[407],
       EE_list[408]: ee_list[408],
       EE_list[409]: ee_list[409],
       EE_list[410]: ee_list[410],
       EE_list[411]: ee_list[411],
       EE_list[412]: ee_list[412],
       EE_list[413]: ee_list[413],
       EE_list[414]: ee_list[414],
       EE_list[415]: ee_list[415],
       EE_list[416]: ee_list[416],
       EE_list[417]: ee_list[417],
       EE_list[418]: ee_list[418],
       EE_list[419]: ee_list[419],
       EE_list[420]: ee_list[420],
       EE_list[421]: ee_list[421],
       EE_list[422]: ee_list[422],
       EE_list[423]: ee_list[423],
       EE_list[424]: ee_list[424],
       EE_list[425]: ee_list[425],
       EE_list[426]: ee_list[426],
       EE_list[427]: ee_list[427],
       EE_list[428]: ee_list[428],
       EE_list[429]: ee_list[429],
       EE_list[430]: ee_list[430],
       EE_list[431]: ee_list[431],
       EE_list[432]: ee_list[432],
       EE_list[433]: ee_list[433],
       EE_list[434]: ee_list[434],
       EE_list[435]: ee_list[435],
       EE_list[436]: ee_list[436],
       EE_list[437]: ee_list[437],
       EE_list[438]: ee_list[438],
       EE_list[439]: ee_list[439],
       EE_list[440]: ee_list[440],
       EE_list[441]: ee_list[441],
       EE_list[442]: ee_list[442],
       EE_list[443]: ee_list[443],
       EE_list[444]: ee_list[444],
       EE_list[445]: ee_list[445],
       EE_list[446]: ee_list[446],
       EE_list[447]: ee_list[447],
       EE_list[448]: ee_list[448],
       EE_list[449]: ee_list[449],
       EE_list[450]: ee_list[450],
       EE_list[451]: ee_list[451],
       EE_list[452]: ee_list[452],
       EE_list[453]: ee_list[453],
       EE_list[454]: ee_list[454],
       EE_list[455]: ee_list[455],
       EE_list[456]: ee_list[456],
       EE_list[457]: ee_list[457],
       EE_list[458]: ee_list[458],
       EE_list[459]: ee_list[459],
       EE_list[460]: ee_list[460],
       EE_list[461]: ee_list[461],
       EE_list[462]: ee_list[462],
       EE_list[463]: ee_list[463],
       EE_list[464]: ee_list[464],
       EE_list[465]: ee_list[465],
       EE_list[466]: ee_list[466],
       EE_list[467]: ee_list[467],
       EE_list[468]: ee_list[468],
       EE_list[469]: ee_list[469],
       EE_list[470]: ee_list[470],
       EE_list[471]: ee_list[471],
       EE_list[472]: ee_list[472],
       EE_list[473]: ee_list[473],
       EE_list[474]: ee_list[474],
       EE_list[475]: ee_list[475],
       EE_list[476]: ee_list[476],
       EE_list[477]: ee_list[477],
       EE_list[478]: ee_list[478],
       EE_list[479]: ee_list[479],
       EE_list[480]: ee_list[480],
       EE_list[481]: ee_list[481],
       EE_list[482]: ee_list[482],
       EE_list[483]: ee_list[483],
       EE_list[484]: ee_list[484],
       EE_list[485]: ee_list[485],
       EE_list[486]: ee_list[486],
       EE_list[487]: ee_list[487],
       EE_list[488]: ee_list[488],
       EE_list[489]: ee_list[489],
       EE_list[490]: ee_list[490],
       EE_list[491]: ee_list[491],
       EE_list[492]: ee_list[492],
       EE_list[493]: ee_list[493],
       EE_list[494]: ee_list[494],
       EE_list[495]: ee_list[495],
       EE_list[496]: ee_list[496],
       EE_list[497]: ee_list[497],
       EE_list[498]: ee_list[498],
       EE_list[499]: ee_list[499],
       E_list[0]: e_list[0],
       E_list[1]: e_list[1],
       E_list[2]: e_list[2],
       E_list[3]: e_list[3],
       E_list[4]: e_list[4],
       E_list[5]: e_list[5],
       E_list[6]: e_list[6],
       E_list[7]: e_list[7],
       E_list[8]: e_list[8],
       E_list[9]: e_list[9],
       E_list[10]: e_list[10],
       E_list[11]: e_list[11],
       E_list[12]: e_list[12],
       E_list[13]: e_list[13],
       E_list[14]: e_list[14],
       E_list[15]: e_list[15],
       E_list[16]: e_list[16],
       E_list[17]: e_list[17],
       E_list[18]: e_list[18],
       E_list[19]: e_list[19],
       E_list[20]: e_list[20],
       E_list[21]: e_list[21],
       E_list[22]: e_list[22],
       E_list[23]: e_list[23],
       E_list[24]: e_list[24],
       E_list[25]: e_list[25],
       E_list[26]: e_list[26],
       E_list[27]: e_list[27],
       E_list[28]: e_list[28],
       E_list[29]: e_list[29],
       E_list[30]: e_list[30],
       E_list[31]: e_list[31],
       E_list[32]: e_list[32],
       E_list[33]: e_list[33],
       E_list[34]: e_list[34],
       E_list[35]: e_list[35],
       E_list[36]: e_list[36],
       E_list[37]: e_list[37],
       E_list[38]: e_list[38],
       E_list[39]: e_list[39],
       E_list[40]: e_list[40],
       E_list[41]: e_list[41],
       E_list[42]: e_list[42],
       E_list[43]: e_list[43],
       E_list[44]: e_list[44],
       E_list[45]: e_list[45],
       E_list[46]: e_list[46],
       E_list[47]: e_list[47],
       E_list[48]: e_list[48],
       E_list[49]: e_list[49],
       E_list[50]: e_list[50],
       E_list[51]: e_list[51],
       E_list[52]: e_list[52],
       E_list[53]: e_list[53],
       E_list[54]: e_list[54],
       E_list[55]: e_list[55],
       E_list[56]: e_list[56],
       E_list[57]: e_list[57],
       E_list[58]: e_list[58],
       E_list[59]: e_list[59],
       E_list[60]: e_list[60],
       E_list[61]: e_list[61],
       E_list[62]: e_list[62],
       E_list[63]: e_list[63],
       E_list[64]: e_list[64],
       E_list[65]: e_list[65],
       E_list[66]: e_list[66],
       E_list[67]: e_list[67],
       E_list[68]: e_list[68],
       E_list[69]: e_list[69],
       E_list[70]: e_list[70],
       E_list[71]: e_list[71],
       E_list[72]: e_list[72],
       E_list[73]: e_list[73],
       E_list[74]: e_list[74],
       E_list[75]: e_list[75],
       E_list[76]: e_list[76],
       E_list[77]: e_list[77],
       E_list[78]: e_list[78],
       E_list[79]: e_list[79],
       E_list[80]: e_list[80],
       E_list[81]: e_list[81],
       E_list[82]: e_list[82],
       E_list[83]: e_list[83],
       E_list[84]: e_list[84],
       E_list[85]: e_list[85],
       E_list[86]: e_list[86],
       E_list[87]: e_list[87],
       E_list[88]: e_list[88],
       E_list[89]: e_list[89],
       E_list[90]: e_list[90],
       E_list[91]: e_list[91],
       E_list[92]: e_list[92],
       E_list[93]: e_list[93],
       E_list[94]: e_list[94],
       E_list[95]: e_list[95],
       E_list[96]: e_list[96],
       E_list[97]: e_list[97],
       E_list[98]: e_list[98],
       E_list[99]: e_list[99],
       E_list[100]: e_list[100],
       E_list[101]: e_list[101],
       E_list[102]: e_list[102],
       E_list[103]: e_list[103],
       E_list[104]: e_list[104],
       E_list[105]: e_list[105],
       E_list[106]: e_list[106],
       E_list[107]: e_list[107],
       E_list[108]: e_list[108],
       E_list[109]: e_list[109],
       E_list[110]: e_list[110],
       E_list[111]: e_list[111],
       E_list[112]: e_list[112],
       E_list[113]: e_list[113],
       E_list[114]: e_list[114],
       E_list[115]: e_list[115],
       E_list[116]: e_list[116],
       E_list[117]: e_list[117],
       E_list[118]: e_list[118],
       E_list[119]: e_list[119],
       E_list[120]: e_list[120],
       E_list[121]: e_list[121],
       E_list[122]: e_list[122],
       E_list[123]: e_list[123],
       E_list[124]: e_list[124],
       E_list[125]: e_list[125],
       E_list[126]: e_list[126],
       E_list[127]: e_list[127],
       E_list[128]: e_list[128],
       E_list[129]: e_list[129],
       E_list[130]: e_list[130],
       E_list[131]: e_list[131],
       E_list[132]: e_list[132],
       E_list[133]: e_list[133],
       E_list[134]: e_list[134],
       E_list[135]: e_list[135],
       E_list[136]: e_list[136],
       E_list[137]: e_list[137],
       E_list[138]: e_list[138],
       E_list[139]: e_list[139],
       E_list[140]: e_list[140],
       E_list[141]: e_list[141],
       E_list[142]: e_list[142],
       E_list[143]: e_list[143],
       E_list[144]: e_list[144],
       E_list[145]: e_list[145],
       E_list[146]: e_list[146],
       E_list[147]: e_list[147],
       E_list[148]: e_list[148],
       E_list[149]: e_list[149],
       E_list[150]: e_list[150],
       E_list[151]: e_list[151],
       E_list[152]: e_list[152],
       E_list[153]: e_list[153],
       E_list[154]: e_list[154],
       E_list[155]: e_list[155],
       E_list[156]: e_list[156],
       E_list[157]: e_list[157],
       E_list[158]: e_list[158],
       E_list[159]: e_list[159],
       E_list[160]: e_list[160],
       E_list[161]: e_list[161],
       E_list[162]: e_list[162],
       E_list[163]: e_list[163],
       E_list[164]: e_list[164],
       E_list[165]: e_list[165],
       E_list[166]: e_list[166],
       E_list[167]: e_list[167],
       E_list[168]: e_list[168],
       E_list[169]: e_list[169],
       E_list[170]: e_list[170],
       E_list[171]: e_list[171],
       E_list[172]: e_list[172],
       E_list[173]: e_list[173],
       E_list[174]: e_list[174],
       E_list[175]: e_list[175],
       E_list[176]: e_list[176],
       E_list[177]: e_list[177],
       E_list[178]: e_list[178],
       E_list[179]: e_list[179],
       E_list[180]: e_list[180],
       E_list[181]: e_list[181],
       E_list[182]: e_list[182],
       E_list[183]: e_list[183],
       E_list[184]: e_list[184],
       E_list[185]: e_list[185],
       E_list[186]: e_list[186],
       E_list[187]: e_list[187],
       E_list[188]: e_list[188],
       E_list[189]: e_list[189],
       E_list[190]: e_list[190],
       E_list[191]: e_list[191],
       E_list[192]: e_list[192],
       E_list[193]: e_list[193],
       E_list[194]: e_list[194],
       E_list[195]: e_list[195],
       E_list[196]: e_list[196],
       E_list[197]: e_list[197],
       E_list[198]: e_list[198],
       E_list[199]: e_list[199],
       E_list[200]: e_list[200],
       E_list[201]: e_list[201],
       E_list[202]: e_list[202],
       E_list[203]: e_list[203],
       E_list[204]: e_list[204],
       E_list[205]: e_list[205],
       E_list[206]: e_list[206],
       E_list[207]: e_list[207],
       E_list[208]: e_list[208],
       E_list[209]: e_list[209],
       E_list[210]: e_list[210],
       E_list[211]: e_list[211],
       E_list[212]: e_list[212],
       E_list[213]: e_list[213],
       E_list[214]: e_list[214],
       E_list[215]: e_list[215],
       E_list[216]: e_list[216],
       E_list[217]: e_list[217],
       E_list[218]: e_list[218],
       E_list[219]: e_list[219],
       E_list[220]: e_list[220],
       E_list[221]: e_list[221],
       E_list[222]: e_list[222],
       E_list[223]: e_list[223],
       E_list[224]: e_list[224],
       E_list[225]: e_list[225],
       E_list[226]: e_list[226],
       E_list[227]: e_list[227],
       E_list[228]: e_list[228],
       E_list[229]: e_list[229],
       E_list[230]: e_list[230],
       E_list[231]: e_list[231],
       E_list[232]: e_list[232],
       E_list[233]: e_list[233],
       E_list[234]: e_list[234],
       E_list[235]: e_list[235],
       E_list[236]: e_list[236],
       E_list[237]: e_list[237],
       E_list[238]: e_list[238],
       E_list[239]: e_list[239],
       E_list[240]: e_list[240],
       E_list[241]: e_list[241],
       E_list[242]: e_list[242],
       E_list[243]: e_list[243],
       E_list[244]: e_list[244],
       E_list[245]: e_list[245],
       E_list[246]: e_list[246],
       E_list[247]: e_list[247],
       E_list[248]: e_list[248],
       E_list[249]: e_list[249],
       E_list[250]: e_list[250],
       E_list[251]: e_list[251],
       E_list[252]: e_list[252],
       E_list[253]: e_list[253],
       E_list[254]: e_list[254],
       E_list[255]: e_list[255],
       E_list[256]: e_list[256],
       E_list[257]: e_list[257],
       E_list[258]: e_list[258],
       E_list[259]: e_list[259],
       E_list[260]: e_list[260],
       E_list[261]: e_list[261],
       E_list[262]: e_list[262],
       E_list[263]: e_list[263],
       E_list[264]: e_list[264],
       E_list[265]: e_list[265],
       E_list[266]: e_list[266],
       E_list[267]: e_list[267],
       E_list[268]: e_list[268],
       E_list[269]: e_list[269],
       E_list[270]: e_list[270],
       E_list[271]: e_list[271],
       E_list[272]: e_list[272],
       E_list[273]: e_list[273],
       E_list[274]: e_list[274],
       E_list[275]: e_list[275],
       E_list[276]: e_list[276],
       E_list[277]: e_list[277],
       E_list[278]: e_list[278],
       E_list[279]: e_list[279],
       E_list[280]: e_list[280],
       E_list[281]: e_list[281],
       E_list[282]: e_list[282],
       E_list[283]: e_list[283],
       E_list[284]: e_list[284],
       E_list[285]: e_list[285],
       E_list[286]: e_list[286],
       E_list[287]: e_list[287],
       E_list[288]: e_list[288],
       E_list[289]: e_list[289],
       E_list[290]: e_list[290],
       E_list[291]: e_list[291],
       E_list[292]: e_list[292],
       E_list[293]: e_list[293],
       E_list[294]: e_list[294],
       E_list[295]: e_list[295],
       E_list[296]: e_list[296],
       E_list[297]: e_list[297],
       E_list[298]: e_list[298],
       E_list[299]: e_list[299],
       E_list[300]: e_list[300],
       E_list[301]: e_list[301],
       E_list[302]: e_list[302],
       E_list[303]: e_list[303],
       E_list[304]: e_list[304],
       E_list[305]: e_list[305],
       E_list[306]: e_list[306],
       E_list[307]: e_list[307],
       E_list[308]: e_list[308],
       E_list[309]: e_list[309],
       E_list[310]: e_list[310],
       E_list[311]: e_list[311],
       E_list[312]: e_list[312],
       E_list[313]: e_list[313],
       E_list[314]: e_list[314],
       E_list[315]: e_list[315],
       E_list[316]: e_list[316],
       E_list[317]: e_list[317],
       E_list[318]: e_list[318],
       E_list[319]: e_list[319],
       E_list[320]: e_list[320],
       E_list[321]: e_list[321],
       E_list[322]: e_list[322],
       E_list[323]: e_list[323],
       E_list[324]: e_list[324],
       E_list[325]: e_list[325],
       E_list[326]: e_list[326],
       E_list[327]: e_list[327],
       E_list[328]: e_list[328],
       E_list[329]: e_list[329],
       E_list[330]: e_list[330],
       E_list[331]: e_list[331],
       E_list[332]: e_list[332],
       E_list[333]: e_list[333],
       E_list[334]: e_list[334],
       E_list[335]: e_list[335],
       E_list[336]: e_list[336],
       E_list[337]: e_list[337],
       E_list[338]: e_list[338],
       E_list[339]: e_list[339],
       E_list[340]: e_list[340],
       E_list[341]: e_list[341],
       E_list[342]: e_list[342],
       E_list[343]: e_list[343],
       E_list[344]: e_list[344],
       E_list[345]: e_list[345],
       E_list[346]: e_list[346],
       E_list[347]: e_list[347],
       E_list[348]: e_list[348],
       E_list[349]: e_list[349],
       E_list[350]: e_list[350],
       E_list[351]: e_list[351],
       E_list[352]: e_list[352],
       E_list[353]: e_list[353],
       E_list[354]: e_list[354],
       E_list[355]: e_list[355],
       E_list[356]: e_list[356],
       E_list[357]: e_list[357],
       E_list[358]: e_list[358],
       E_list[359]: e_list[359],
       E_list[360]: e_list[360],
       E_list[361]: e_list[361],
       E_list[362]: e_list[362],
       E_list[363]: e_list[363],
       E_list[364]: e_list[364],
       E_list[365]: e_list[365],
       E_list[366]: e_list[366],
       E_list[367]: e_list[367],
       E_list[368]: e_list[368],
       E_list[369]: e_list[369],
       E_list[370]: e_list[370],
       E_list[371]: e_list[371],
       E_list[372]: e_list[372],
       E_list[373]: e_list[373],
       E_list[374]: e_list[374],
       E_list[375]: e_list[375],
       E_list[376]: e_list[376],
       E_list[377]: e_list[377],
       E_list[378]: e_list[378],
       E_list[379]: e_list[379],
       E_list[380]: e_list[380],
       E_list[381]: e_list[381],
       E_list[382]: e_list[382],
       E_list[383]: e_list[383],
       E_list[384]: e_list[384],
       E_list[385]: e_list[385],
       E_list[386]: e_list[386],
       E_list[387]: e_list[387],
       E_list[388]: e_list[388],
       E_list[389]: e_list[389],
       E_list[390]: e_list[390],
       E_list[391]: e_list[391],
       E_list[392]: e_list[392],
       E_list[393]: e_list[393],
       E_list[394]: e_list[394],
       E_list[395]: e_list[395],
       E_list[396]: e_list[396],
       E_list[397]: e_list[397],
       E_list[398]: e_list[398],
       E_list[399]: e_list[399],
       E_list[400]: e_list[400],
       E_list[401]: e_list[401],
       E_list[402]: e_list[402],
       E_list[403]: e_list[403],
       E_list[404]: e_list[404],
       E_list[405]: e_list[405],
       E_list[406]: e_list[406],
       E_list[407]: e_list[407],
       E_list[408]: e_list[408],
       E_list[409]: e_list[409],
       E_list[410]: e_list[410],
       E_list[411]: e_list[411],
       E_list[412]: e_list[412],
       E_list[413]: e_list[413],
       E_list[414]: e_list[414],
       E_list[415]: e_list[415],
       E_list[416]: e_list[416],
       E_list[417]: e_list[417],
       E_list[418]: e_list[418],
       E_list[419]: e_list[419],
       E_list[420]: e_list[420],
       E_list[421]: e_list[421],
       E_list[422]: e_list[422],
       E_list[423]: e_list[423],
       E_list[424]: e_list[424],
       E_list[425]: e_list[425],
       E_list[426]: e_list[426],
       E_list[427]: e_list[427],
       E_list[428]: e_list[428],
       E_list[429]: e_list[429],
       E_list[430]: e_list[430],
       E_list[431]: e_list[431],
       E_list[432]: e_list[432],
       E_list[433]: e_list[433],
       E_list[434]: e_list[434],
       E_list[435]: e_list[435],
       E_list[436]: e_list[436],
       E_list[437]: e_list[437],
       E_list[438]: e_list[438],
       E_list[439]: e_list[439],
       E_list[440]: e_list[440],
       E_list[441]: e_list[441],
       E_list[442]: e_list[442],
       E_list[443]: e_list[443],
       E_list[444]: e_list[444],
       E_list[445]: e_list[445],
       E_list[446]: e_list[446],
       E_list[447]: e_list[447],
       E_list[448]: e_list[448],
       E_list[449]: e_list[449],
       E_list[450]: e_list[450],
       E_list[451]: e_list[451],
       E_list[452]: e_list[452],
       E_list[453]: e_list[453],
       E_list[454]: e_list[454],
       E_list[455]: e_list[455],
       E_list[456]: e_list[456],
       E_list[457]: e_list[457],
       E_list[458]: e_list[458],
       E_list[459]: e_list[459],
       E_list[460]: e_list[460],
       E_list[461]: e_list[461],
       E_list[462]: e_list[462],
       E_list[463]: e_list[463],
       E_list[464]: e_list[464],
       E_list[465]: e_list[465],
       E_list[466]: e_list[466],
       E_list[467]: e_list[467],
       E_list[468]: e_list[468],
       E_list[469]: e_list[469],
       E_list[470]: e_list[470],
       E_list[471]: e_list[471],
       E_list[472]: e_list[472],
       E_list[473]: e_list[473],
       E_list[474]: e_list[474],
       E_list[475]: e_list[475],
       E_list[476]: e_list[476],
       E_list[477]: e_list[477],
       E_list[478]: e_list[478],
       E_list[479]: e_list[479],
       E_list[480]: e_list[480],
       E_list[481]: e_list[481],
       E_list[482]: e_list[482],
       E_list[483]: e_list[483],
       E_list[484]: e_list[484],
       E_list[485]: e_list[485],
       E_list[486]: e_list[486],
       E_list[487]: e_list[487],
       E_list[488]: e_list[488],
       E_list[489]: e_list[489],
       E_list[490]: e_list[490],
       E_list[491]: e_list[491],
       E_list[492]: e_list[492],
       E_list[493]: e_list[493],
       E_list[494]: e_list[494],
       E_list[495]: e_list[495],
       E_list[496]: e_list[496],
       E_list[497]: e_list[497],
       E_list[498]: e_list[498],
       E_list[499]: e_list[499]
   }))

'''