

import pandas as pd
import numpy as np
import tensorflow as tf
from tre_simple import *
from parser import *
from datetime import datetime

N_HIDDEN_STATES = 3
N_SYMBOLS = 367
MAX_CHILD = 32


epoche =1


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
    sum = tf.expand_dims(sum, axe)
    sum = tf.tile(sum, [shape2,1,1])
    rand_sum_one = tf.divide(rand, sum)

    return rand_sum_one




#|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||start||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
print("leggo il dataset")
data_set = dataset_parser()

print("parto")
start = datetime.now()

n=0

#N_HIDDEN_STATES da 2 a 20 non di più, va calcolato l'algoritmo per i vari valori, che fanno cambiare il tutto di molto IMPORTANTE


scope_tree = "scope_n0"
scope_epoca = "epoca_n0"
#nel caso non vengano passati dei valori iniziali ai parametri essi venono inizializati random
pi = random_sum_one2(0, N_HIDDEN_STATES, MAX_CHILD)
sp_p = random_sum_one1(MAX_CHILD)
A = random_sum_one3(0, N_HIDDEN_STATES, N_HIDDEN_STATES, MAX_CHILD)
bi = random_sum_one2(1, N_HIDDEN_STATES, N_SYMBOLS)




#per il numero delle epoco eseguo l'E-M

for i in range(0, epoche):
    print("EPOCA: ",i)
    scope_epoca = scope_epoca[:-len(str(i - 1))] + str(i)
    with tf.variable_scope(scope_epoca):
        #eseguo E-STEP per ogni albero nel dataset
        var_EE_list = []
        var_E_list = []
        for j in range(0,len(data_set)):
            print(data_set[j])
            print(data_set[j].size)
            print(len(data_set[j].struct[-1]))
            scope_tree=scope_tree[:-len(str(j-1))]+str(j)
            with tf.variable_scope(scope_tree):
                # |||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||Reversed_Upward_Downward||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
                t= data_set[j]
                # upward parameters beta
                up_ward = np.zeros((t.size, N_HIDDEN_STATES))
                a_up_ward = np.zeros((t.size, N_HIDDEN_STATES, MAX_CHILD))

                # internal node prior
                in_prior = np.ones((N_HIDDEN_STATES, t.size))

                # pairwise smoothed posterior
                E = np.zeros((t.size-1,N_HIDDEN_STATES))
                EE = np.zeros((t.size, N_HIDDEN_STATES, N_HIDDEN_STATES))

                ph_post = tf.placeholder(shape=[t.size, N_HIDDEN_STATES], dtype=tf.float64)
                ph_s_post = tf.placeholder(shape=[MAX_CHILD, N_HIDDEN_STATES, N_HIDDEN_STATES, t.size],
                                           dtype=tf.float64)
                ph_in_prior = tf.placeholder(shape=[N_HIDDEN_STATES, t.size], dtype=tf.float64)

                for ii in range(0, N_HIDDEN_STATES):
                    #for jj in range(t.size - len(t.struct[-1]), t.size):
                    for jj in range(t.size):
                        in_prior[ii, jj] = 1/N_HIDDEN_STATES
                print(in_prior)
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


                # |||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||compute_17||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
                label = []
                posizione = []
                for node in t.struct[-1]:
                    label.append(node.label)
                    posizione.append(node.pos)
                #prelevo da bi e pi i dati relativi alle foglie
                aux1 = tf.gather(bi, label, axis=1)
                aux2 = tf.gather(pi, posizione, axis=1)

                nume = tf.multiply(aux1, aux2)  # Element-wise multiplication
                #den = tf.einsum('ij,ji->i', tf.transpose(aux1), aux2)  # Einstein summation per moltiplicazione di
                den = tf.reduce_sum(nume,[0])
                # righe e colonne con lo stesso indice

                ris_17_t = tf.divide(nume, den)
                ris_17_t = tf.transpose(ris_17_t, perm=[1, 0])

                head = tf.slice(var_up_ward, [0, 0], [t.struct[-1][0].name, N_HIDDEN_STATES])
                var_up_ward = tf.concat([head, ris_17_t], 0)

                # |||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||compute_internal_node_prior||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
                print("sp_p-----------------------------------------------",sp_p)
                print("A---------------------------------------------------",A)

                aux1 = tf.multiply(sp_p, A)  # broadcast implicito
                print("aux1------------------------------------------------",aux1)


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

                    print(nomi_figli)
                    aux2 = tf.gather(var_in_prior, nomi_figli, axis=1)
                   #aux2 = tf.where(tf.is_zero(aux2), tf.zeros_like(aux2), aux2)

                    aux2 = tf.transpose(aux2, perm=[1, 0, 2])

                    aux2 = tf.expand_dims(aux2, 1)

                    aux2 = tf.tile(aux2, [1, N_HIDDEN_STATES, 1, 1])

                    #  di figli di un nodo  in un unica matrice N_HIDDEN_STATES*L*(numero di nodi del livello)
                    # qui moltiplicazione
                    # questa è una serie di matrici, tante quanti sono i nodi del livello esaminati

                    aux3 = tf.multiply(aux1, aux2)

                    s = tf.reduce_sum(aux3, [2, 3])
                    s = tf.transpose(s)

                    # prelevo i valori iniziali e quelli finali che non devono essere aggiornati in questo ciclo
                    head = tf.slice(var_in_prior, [0, 0], [N_HIDDEN_STATES,t.struct[i][-1].name + 1 - int((s.shape[1]))])  # ricorda che questa deriva da quella sopra
                    # [N_HIDDEN_STATES, (t.size  -(t.size - t.struct[i][-1].name - 1) - int((s.shape[1])) )])

                    tail = tf.slice(var_in_prior, [0, t.struct[i][-1].name + 1],
                                    [N_HIDDEN_STATES,
                                     t.size - t.struct[i][-1].name - 1])  # potrei farlo anche con un constant

                    var_in_prior = tf.concat([head, s, tail], 1)  # aggiorno i nuovi valore trovati


                print("var_in_prior.-------",var_in_prior)
                test_var_in_prior = tf.reduce_sum(var_in_prior,[0])


                # |||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||up step||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
                # DDD non dovrebbe fare uno la somma dei 21? sulle i?

                # up step
                for i in range(len(t.struct) - 2, -1  , -1):
                    # |||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||compute_21||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
                    nomi_figli = []
                    nomi_nodi  = []
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


                    numerator_n = tf.multiply(A, aux_up_ward)

                    numerator = tf.reduce_sum(numerator_n, [2])


                    node_in_priors = tf.gather(var_in_prior, nomi_nodi, axis=1)


                    # faccio un broadcast esplicito duplicando la matrice su una nuova dimenzione  per il numero di stati nascosti
                    node_in_priors = tf.expand_dims(node_in_priors, 0)
                    node_in_priors = tf.tile(node_in_priors, [MAX_CHILD, 1, 1])
                    node_in_priors = tf.transpose(node_in_priors, perm=[2, 1, 0])
                    s = tf.divide(numerator, node_in_priors)


                    head = tf.slice(var_a_up_ward, [0, 0, 0],
                                    [t.struct[i][-1].name + 1 - int((s.shape[0])), N_HIDDEN_STATES,
                                     MAX_CHILD])  # potrei farlo anche con un constant

                    tail = tf.slice(var_a_up_ward, [t.struct[i][-1].name + 1, 0, 0],
                                    [t.size - t.struct[i][-1].name - 1, N_HIDDEN_STATES,
                                     MAX_CHILD])  # potrei farlo anche con un constant


                    var_a_up_ward = tf.concat([head, s, tail], 0)

                    test_a_var_up_ward = tf.reduce_sum(var_a_up_ward,[1])
                    # |||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||compute_19||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||

                    sli_var_in_prior = tf.ones([N_HIDDEN_STATES, 0], tf.float64)

                    labels = []
                    nomi = []
                    for node in t.struct[i]:
                        labels.append(node.label)
                        nomi.append(node.name)

                    sli_ph_bi = tf.gather(bi, labels, axis=1)

                    #second_term = tf.gather(var_a_up_ward, nomi)    ## è sempre la s di sopra???
                    #considerando la semplificazione non devo prendere l'intero var_a_up_ward ma solo la sua parte superiore che preleviamo da sopra
                    second_term=numerator

                    first_term = tf.expand_dims(sp_p, 0)  # faccio un broadcast esplicito duplicando la matrice su una nuova dimenzione  per il numero di stati nascosti
                    first_term = tf.expand_dims(first_term, 0)
                    first_term = tf.tile(first_term, [len(t.struct[i]), N_HIDDEN_STATES, 1])

                    # per in num
                    somm = tf.reduce_sum(tf.multiply(first_term, second_term), [2])  # sommo sulla dim 2 (sommo le righe)

                    sli_ph_bi = tf.transpose(sli_ph_bi, perm=[1, 0])

                    numerator = tf.multiply(sli_ph_bi, somm)

                    # per il den
                    bb = tf.expand_dims(sli_ph_bi, 2)
                    bb = tf.tile(bb, [1, 1, MAX_CHILD])

                    denominator_int = tf.reduce_sum(tf.multiply(tf.multiply(bb, first_term), second_term),
                                                    [2, 1])  # sommo sulla dim 2 (sommo le righe)

                    denominator = tf.expand_dims(denominator_int, 1)

                    # duplico il denominatore per ogni stato nascosto di un nodo
                    denominator = tf.tile(denominator, [1,N_HIDDEN_STATES])

                    # finale
                    ris_19 = tf.divide(numerator, denominator)


                    head = tf.slice(var_up_ward, [0, 0], [t.struct[i][-1].name + 1 - int((ris_19.shape[0])), N_HIDDEN_STATES])
                    tail = tf.slice(var_up_ward, [t.struct[i][-1].name +1 , 0], [t.size - t.struct[i][-1].name - 1, N_HIDDEN_STATES])
                    # tail=tf.slice(var_up_ward, [t.struct[i][-1].name -1 , 0],   per far compilare



                    var_up_ward = tf.concat([head, ris_19, tail], 0)


                test_var_up_ward1 = tf.reduce_sum(var_up_ward, [1])
                test_var_up_ward2 = tf.reduce_sum(var_up_ward, [0])

                # |||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||    set_base_case      ||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||

                base_case = tf.slice(var_up_ward, [0, 0], [1, N_HIDDEN_STATES])
                var_E = tf.concat([base_case, var_E], 0)


                # |||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||        # down step            ||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||

                # down step
                #for i in range(1, 2):
                for i in range(1, len(t.struct)):
                    # |||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||compute_24||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||

                    padri = []
                    posizione = []
                    nomi_nodi = []
                    for node in t.struct[i]:
                        nomi_nodi.append(node.name)
                        padri.append(node.father.name)
                        posizione.append(node.pos)

                    sli_E = tf.gather(var_E, padri)
                    sli_up_ward = tf.gather(var_up_ward, nomi_nodi)
                    sli_sp_p = tf.gather(sp_p, posizione)
                    sli_A = tf.gather(A, posizione, axis=2)
                    sli_A = tf.transpose(sli_A, perm=[2, 0, 1])

                    #per il den
                    sli_in_prior = tf.gather(var_in_prior, padri, axis=1)
                    sli_var_a_up_ward = tf.gather(var_a_up_ward, padri)

                    # per il numeratore
                    sli_E = tf.expand_dims(sli_E, 1)
                    sli_E = tf.tile(sli_E, [ 1,N_HIDDEN_STATES, 1])


                    sli_up_ward = tf.expand_dims(sli_up_ward, 1)
                    sli_up_ward = tf.tile(sli_up_ward, [1, N_HIDDEN_STATES, 1])

                    sli_sp_p = tf.expand_dims(sli_sp_p, 1)
                    sli_sp_p = tf.expand_dims(sli_sp_p, 1)
                    sli_sp_p = tf.tile(sli_sp_p, [1, N_HIDDEN_STATES, N_HIDDEN_STATES])


                    numerator = tf.multiply(sli_E, sli_up_ward)
                    numerator = tf.multiply(numerator, sli_sp_p)
                    numerator = tf.multiply(numerator, sli_A)


                    # per il denominatore

                    a_sp_p = tf.expand_dims(sp_p, 0)
                    a_sp_p = tf.expand_dims(a_sp_p, 0)
                    a_sp_p = tf.tile(a_sp_p, [len(t.struct[i]), N_HIDDEN_STATES, 1])

                    to_sum = tf.multiply(a_sp_p, sli_var_a_up_ward)
                    added = tf.reduce_sum(to_sum, [2])  # sommo nella dim 2

                    sli_in_prior = tf.transpose(sli_in_prior, perm=[1, 0])

                    denominator = tf.multiply(sli_in_prior, added)
                    denominator = tf.expand_dims(denominator, 1)
                    denominator = tf.tile(denominator, [1, N_HIDDEN_STATES, 1])

                    ris_24 = tf.divide(numerator, denominator)
                    #uniform = tf.reduce_sum(ris_24, [1])
                    #uniform = tf.expand_dims(uniform, 1)
                    #uniform = tf.tile(uniform, [1, N_HIDDEN_STATES,1])
                    #ris_24 = tf.divide(ris_24, uniform)
                    #DDD

                    # |||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||inglobe_ris_liv||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||

                    head = tf.slice(var_EE, [0, 0, 0],
                                    [t.struct[i][-1].name + 1 - int((ris_24.shape[0])), N_HIDDEN_STATES,
                                     N_HIDDEN_STATES])
                    tail = tf.slice(var_EE, [t.struct[i][-1].name + 1, 0, 0],
                                    [t.size - t.struct[i][-1].name - 1, N_HIDDEN_STATES, N_HIDDEN_STATES])

                    var_EE = tf.concat([head, ris_24, tail], 0)



                    # |||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||computer25||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||

                    ris_25 = tf.reduce_sum(ris_24, [2])


                    #qui faccio in modo che la somma di var_E  sugli stati nascosti sia uguale ad 1------------------------------------------------------- è corretto?
                    uniform = tf.reduce_sum(ris_25, [1])
                    uniform = tf.expand_dims(uniform, 1)
                    uniform = tf.tile(uniform, [1, N_HIDDEN_STATES])
                    ris_25 = tf.divide(ris_25, uniform)

                    head = tf.slice(var_E, [0, 0], [t.struct[i][-1].name + 1 - int((ris_24.shape[0])),
                                                    N_HIDDEN_STATES])  # _________________-da controllare la dim giusta in shape
                    tail = tf.slice(var_E, [t.struct[i][-1].name + 1, 0],
                                    [t.size - t.struct[i][-1].name - 1, N_HIDDEN_STATES])

                    var_E = tf.concat([head, ris_25, tail], 0)

                    test_var_E = tf.reduce_sum(var_E, [1])
                    # |||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||Reversed_Upward_Downward||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
                #uniform = tf.reduce_sum(var_EE, [1])
                #uniform = tf.expand_dims(uniform, 1)
                #uniform = tf.tile(uniform, [1, N_HIDDEN_STATES, 1])
                #var_EE = tf.divide(var_EE, uniform)
                #var_EE = tf.where(tf.is_nan(var_EE), tf.zeros_like(var_EE),var_EE)


                uniform = tf.reduce_sum(var_E, [1])
                uniform = tf.expand_dims(uniform, 1)
                uniform = tf.tile(uniform, [1, N_HIDDEN_STATES])
                var_E = tf.divide(var_E, uniform)


                var_EE_list.append(var_EE)
                var_E_list.append(var_E)
        print("         E-step")


        # |||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||M_step||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||

        lista_prior = []
        lista_n_in = []
        aux_list_prior = []
        aux_list_sp = []
        n_l_list = np.zeros(MAX_CHILD)
        n_ii_list = np.zeros(MAX_CHILD)
        sum_N_I = 0
        max_l = -1
        num_tm_yu_list = []
        den_tm_yu_list = []

        # MULTINOMIAL
        # calcolo Tm_Yu
        for i in range(0, len(var_EE_list)):
            tm_yu = np.zeros([int(data_set[i].size), N_SYMBOLS])
            for level in data_set[i].struct:
                for node in level:
                    tm_yu[node.name, node.label] = 1

            e_resized = tf.expand_dims(var_E_list[i], 2)
            e_resized = tf.tile(e_resized, [1, 1, N_SYMBOLS])

            resized_tm_yu = tf.expand_dims(tm_yu, 2)  # qui posso portare fuori dal for per ottimizzare...
            resized_tm_yu = tf.tile(resized_tm_yu, [1, 1, N_HIDDEN_STATES])
            resized_tm_yu = tf.transpose(resized_tm_yu, perm=[0, 2, 1])

            numerator_tm_yu = tf.multiply(resized_tm_yu, e_resized)
            numerator_tm_yu = tf.reduce_sum(numerator_tm_yu, [0])

            denominator_tm_yu = tf.reduce_sum(var_E_list[i], [0])
            denominator_tm_yu = tf.expand_dims(denominator_tm_yu, 1)
            denominator_tm_yu = tf.tile(denominator_tm_yu, [1, N_SYMBOLS]) # queste posso tirarle fuori...

            num_tm_yu_list.append(numerator_tm_yu)
            den_tm_yu_list.append(denominator_tm_yu)

        numerator_tm_yu = tf.stack(num_tm_yu_list, 0)
        denominator_tm_yu = tf.stack(den_tm_yu_list, 0)

        numerator_tm_yu = tf.reduce_sum(numerator_tm_yu, [0])
        denominator_tm_yu = tf.reduce_sum(denominator_tm_yu, [0])
        result_multinomial = tf.divide(numerator_tm_yu, denominator_tm_yu)
        result_multinomial = tf.where(tf.is_inf(result_multinomial), tf.zeros_like(result_multinomial), result_multinomial)
        result_multinomial = tf.where(tf.is_nan(result_multinomial), tf.zeros_like(result_multinomial), result_multinomial)

        # calcolo il numero totale di nodi nell L-esima posizione
        # e anche il numero massimo di nodi nella l-esima posizione, così da poter dimenzionare in maniera opportuna le dim
        for i in range(0, len(data_set)):
            #aggiungo una lista per ogni albero
            lista_prior.append([])
            lista_n_in.append([])
            n_l_list = n_l_list + data_set[i].N_L
            n_ii_list = n_ii_list + data_set[i].N_II
            sum_N_I = sum_N_I + data_set[i].N_I
            for l_number in data_set[i].N_L:
                if max_l < l_number:
                    max_l = l_number

            for j in range(0, MAX_CHILD):
                #aggiungo una lista per ogni posibile posizione di figlio
                lista_prior[i].append([])
                lista_n_in[i].append([])

        for i in range(0, len(data_set)):

            # SP & STATE TRANSICTION
            for jj in range(0, len(data_set[i].struct) - 1):
                for nodo in data_set[i].struct[jj]:
                    for kk in range(0, len(nodo.children)):
                        lista_n_in[i][kk].append(nodo.children[kk].name)
            # uniformo la lunghezza così da non rendere il tensore sparso per la futura gather
            for in_list in lista_n_in[i]:
                start = len(in_list)
                for k in range(start, int(max_l + 1)):
                    in_list.append(0)

            slice = tf.gather(var_EE_list[i], lista_n_in[i])
            aux_list_sp.append(slice)

            # prior
            for nodo in (data_set[i].struct[-1]):
                lista_prior[i][nodo.pos-1].append(nodo.name)


            j = 0
            for in_list in lista_prior[i]:
                start = len(in_list)
                for k in range(start, int(max_l)):
                    in_list.append(0)
            slice = tf.gather(var_EE_list[i], lista_prior[i])
            aux_list_prior.append(slice)

        # SP
        aux = tf.stack(aux_list_sp, 0)
        summed_sp = tf.reduce_sum(aux, [4, 3, 2, 0])
        summed_sp2 = tf.reduce_sum(aux, [4, 3])

        result_sp = tf.divide(summed_sp, sum_N_I*MAX_CHILD)
        #result_sp = tf.divide(summed_sp, den) #DDD se non è così elimina n_ii_list e tutto quello collegato precedentemente

        result_sp = tf.where(tf.is_inf(result_sp), tf.zeros_like(result_sp), result_sp)
        result_sp = tf.where(tf.is_nan(result_sp), tf.zeros_like(result_sp), result_sp)

        # STATE TRANSICTION (A)
        numerator_stat_tran = tf.reduce_sum(aux, [2,0])
        denominator_stat_tran = tf.reduce_sum(aux, [3, 2, 0])
        denominator_stat_tran = tf.expand_dims(denominator_stat_tran, 2)

        denominator_stat_tran = tf.tile(denominator_stat_tran, [1, 1, N_HIDDEN_STATES])
        result_state_trans = tf.divide(numerator_stat_tran, denominator_stat_tran)
        result_state_trans = tf.transpose(result_state_trans, [2, 1, 0])
        result_state_trans = tf.where(tf.is_inf(result_state_trans), tf.zeros_like(result_state_trans), result_state_trans)
        result_state_trans = tf.where(tf.is_nan(result_state_trans), tf.zeros_like(result_state_trans), result_state_trans)

        # PRIOR
        aux = tf.stack(aux_list_prior, 0)
        summed_prior = tf.reduce_sum(aux, [4, 2, 0])#DDD
        n_l_list = tf.expand_dims(n_l_list, 1)
        n_l_list = tf.tile(n_l_list, [1, N_HIDDEN_STATES])
        result_prior = tf.divide(summed_prior, n_l_list)
        result_prior = tf.where(tf.is_inf(result_prior), tf.zeros_like(result_prior), result_prior)
        result_prior = tf.where(tf.is_nan(result_prior), tf.zeros_like(result_prior), result_prior)
        result_prior = tf.transpose(result_prior, [1, 0])

        #DDD devo far si che la somma suglii ia uno?

        new_pi=result_prior
        new_sp_p=result_sp
        new_A=result_state_trans
        new_bi=result_multinomial
        # |||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||       fine        M_step      ||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
        print("         M-step")

        pi = new_pi
        sp_p = new_sp_p
        A = new_A
        bi = new_bi


        #t_pi = random_sum_one2(1, N_HIDDEN_STATES, MAX_CHILD)
        #t_sp_p = random_sum_one1(MAX_CHILD)
        #t_A = random_sum_one3(0, N_HIDDEN_STATES, N_HIDDEN_STATES, MAX_CHILD)
        #t_bi = random_sum_one2(0, N_HIDDEN_STATES, N_SYMBOLS)
        # |||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||M_stlog_likelihoodep||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
        '''
        tot = 0
        for i in range(0, len(data_set)):

            # prelevo i nodi interni e foglia
            leaf_node = tf.slice(var_E_list[i], [data_set[i].struct[-1][0].name, 0],
                                 [data_set[i].size - data_set[i].struct[-1][0].name, N_HIDDEN_STATES])

            # prima sommatoria
            # salvo e prelevo la lista dell'indicatore posizionale di ogni nodo foglia
            posizione_foglie = []
            for node in data_set[i].struct[-1]:
                posizione_foglie.append(node.pos)
            log_pi = tf.gather(pi, posizione_foglie, axis=1)
            log_pi = tf.log(log_pi)
            log_pi = tf.transpose(log_pi, [1, 0])
            log_pi = tf.where(tf.is_inf(log_pi), tf.zeros_like(log_pi), log_pi)
            log_pi = tf.cast(log_pi, tf.float64)

            prima_somm = tf.multiply(log_pi, leaf_node)
            prima_somm = tf.reduce_sum(prima_somm, [0, 1])

            # seconda sommatoria
            label_nodi = []
            for level in data_set[i].struct:
                for node in level:
                    label_nodi.append(node.label)
            log_bi = tf.gather(bi, label_nodi, axis=1)
            log_bi = tf.log(log_bi)
            log_bi = tf.transpose(log_bi, [1, 0])
            log_bi = tf.cast(log_bi, tf.float64)
            seconda_somm = tf.multiply(log_bi, var_E_list[i])
            seconda_somm = tf.reduce_sum(seconda_somm, [0, 1])

            # terza sommatoria

            posizione_nodi_interni = []
            for j in range(0, data_set[i].struct[-1][0].name):
                posizione_nodi_interni.append([])

            internal_node_ee = tf.slice(var_EE_list[i], [0, 0, 0],
                                        [data_set[i].struct[-1][0].name, N_HIDDEN_STATES, N_HIDDEN_STATES])
            internal_node_ee = tf.reduce_sum(internal_node_ee, [2, 1])
            for level in data_set[i].struct[:-1]:
                for node in level:
                    for child in node.children:
                        posizione_nodi_interni[node.name].append(child.name)

            for in_list in posizione_nodi_interni:
                start = len(in_list)
                for k in range(start, MAX_CHILD):
                    in_list.append(0)

            ee_sum_c_c = tf.reduce_sum(var_EE_list[i], [2, 1])

            psul = tf.gather(ee_sum_c_c, posizione_nodi_interni)

            log_sp_p = tf.log(sp_p)

            log_sp_p = tf.where(tf.is_inf(log_sp_p), tf.zeros_like(log_sp_p), log_sp_p)

            log_sp_p = tf.cast(log_sp_p, tf.float64)
            terza_somm = tf.multiply(psul, log_sp_p)
            terza_somm = tf.reduce_sum(terza_somm, [0, 1])

            # QUARTA SOMMATORIA
            pqqsy = tf.gather(var_EE_list[i], posizione_nodi_interni)

            log_A = tf.transpose(A, [2, 1, 0])
            log_A = tf.log(log_A)
            log_A = tf.where(tf.is_nan(log_A), tf.zeros_like(log_A), log_A)
            log_A = tf.cast(log_A, tf.float64)
            quarta_somm = tf.multiply(pqqsy,
                                      log_A)  # ________________________________________________________indicie ij da controllare
            quarta_somm = tf.reduce_sum(quarta_somm, [1, 2, 3, 0])

            tot = tot + prima_somm + seconda_somm + terza_somm + quarta_somm


            # |||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||           tlog_likelihood         fine            ||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
        # |||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||   fine            M_stlog_likelihoodep||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
'''
print("pi------------->",pi)
print("sp_p------------->",sp_p)
print("A------------->",A)
print("bi------------->",bi)
print("\n\n")
t_p=tf.reduce_sum(pi,[0])
t_sp=tf.reduce_sum(sp_p,[0])
t_a=tf.reduce_sum(A,[0])
t_bi=tf.reduce_sum(bi,[1])
with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    # print(sess.run([aux,  pi,t_pi,  sp_p,t_sp_p,  A,t_A,  bi,t_bi]))
    print(sess.run([t_p,t_sp,t_a,t_bi]))

#||||||||||||||||||||||||||||||||||||||||||||||LOGLIKEHOLD||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||


#||||||||||||||||||||||||||||||||||||||||||||||||||||||||E-STEP||||||||||||||||||||||||||||||||||||||||||||||||||||||||

