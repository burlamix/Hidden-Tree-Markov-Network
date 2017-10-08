import numpy as np
import tensorflow as tf
from tre_simple import *
from parser import *
from datetime import datetime





def training(dataset,epoche,pi=None,sp_p=None,A=None,bi=None):

    n=0

    #N_HIDDEN_STATES da 2 a 20 non di più, va calcolato l'algoritmo per i vari valori, che fanno cambiare il tutto di molto IMPORTANTE
    N_HIDDEN_STATES=3

    MAX_CHILD=33
    N_SYMBOLS=366

    scope_tree = "scope_n0"
    t = dataset_parser()
    var_EE_list=[]
    var_E_list =[]
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

        #eseguo E-STEP per ogni albero nel dataset
        start = datetime.now()

        for j in range(0,len(dataset)):

            scope_tree=scope_tree[:-len(str(j-1))]+str(j)
            with tf.variable_scope(scope_tree):
                start_2 = datetime.now()
                (var_EE, var_E) = Reversed_Upward_Downward(sp_p, A, bi, pi, N_HIDDEN_STATES, MAX_CHILD, dataset[j])
                print(var_EE)
                var_EE_list.append(var_EE)
                var_E_list.append(var_E)
                print(scope_tree, (datetime.now() - start_2).total_seconds())

            # dal quale mi salvo var_EE in una lista
        print("TEMPO TOTALE : ",(datetime.now()-start).total_seconds() )

        #eseguo M-STEP
        #uso la lista di var_EE calcolata in precedenza

    return var_EE_list,var_E_list


#||||||||||||||||||||||||||||||||||||||||||||||||||||||||E-STEP||||||||||||||||||||||||||||||||||||||||||||||||||||||||

def Reversed_Upward_Downward(ph_sp_p, ph_A, ph_bi, ph_pi, N_HIDDEN_STATES, MAX_CHILD, t):

    #start_all = datetime.now()
    #start = datetime.now()

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

    #print("var in : ",(datetime.now()-start).total_seconds() * 1000)

###########################################################################################################################

    # start = datetime.now()
    var_up_ward = compute_17(ph_bi,ph_pi,var_up_ward,N_HIDDEN_STATES,t)
    # print("17    :  ",(datetime.now()-start).total_seconds() * 1000)

    # start = datetime.now()
    var_in_prior = compute_internal_node_prior(var_in_prior, ph_sp_p, ph_A, t, N_HIDDEN_STATES,MAX_CHILD)
    # print("prior :  ",(datetime.now()-start).total_seconds() * 1000)

    #up step
   # start = datetime.now()
    for i in range(len(t.struct) - 2, -1, -1):

        var_a_up_ward = compute_21(ph_A,var_in_prior,var_a_up_ward,var_up_ward,i,t,N_HIDDEN_STATES,MAX_CHILD)
        var_up_ward = compute_19(ph_bi, ph_sp_p, var_a_up_ward, var_in_prior, var_up_ward, t,i, N_HIDDEN_STATES, MAX_CHILD)

    var_E = set_base_case(var_up_ward,var_E,t,N_HIDDEN_STATES)
    #print("up_ste : ",(datetime.now()-start).total_seconds() * 1000)

    # start = datetime.now()
    # down step
    for i in range(1, len(t.struct)  ): # il -1 l'ho iserito per far compilare e non sono sicuro sia corretto___________________________________________cambialo devi fare anche le foglie______________

        #start1 = datetime.now()
        ris_24 = compute_24(ph_sp_p, ph_A,var_E, var_EE, var_up_ward, var_in_prior, var_a_up_ward, t, i, N_HIDDEN_STATES, MAX_CHILD)
        #print("                     compute_24 : ", (datetime.now() - start1).total_seconds() * 1000)

        #start1 = datetime.now()
        var_EE = inglobe_ris_liv(ris_24, var_EE, t, i,  N_HIDDEN_STATES)
        #print("                     inglobe_ris_liv : ", (datetime.now() - start1).total_seconds() * 1000)

        #start1 = datetime.now()
        var_E =  compute_25(ris_24, var_E, i, t, N_HIDDEN_STATES)
        #print("                     compute_25 : ", (datetime.now() - start1).total_seconds() * 1000)
        #print("\n")


    #print("do_ste : ",(datetime.now()-start).total_seconds() * 1000)

    #print("TOTALE:  ",(datetime.now()-start_all).total_seconds() * 1000)

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
MAX_CHILD=32


def M_step(EE_list,data_set):

    lista_prior = []
    lista_n_in = []
    aux_list_prior = []
    aux_list_sp = []
    n_l_list = np.zeros(MAX_CHILD)
    sum_N_I =0
    max_l =-1
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

    #for i in range(0,len(data_set)):
    for i in range(0,500):




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
    print("aux",aux)
    summed_sp = tf.reduce_sum(aux,[3,2,1,0])#-------------------------------------------------------------CONTROLLARE L'INDICE I,J
    result_sp = tf.divide(summed_sp,sum_N_I)

    # STATE TRANSICTION
    numerator_stat_tran = tf.reduce_sum(aux,[2,0])#-------------------------------------------------------------CONTROLLARE L'INDICE I,J
    denominator_stat_tran = tf.reduce_sum(aux,[4,2,0])#-------------------------------------------------------------CONTROLLARE L'INDICE I,J
    denominator_stat_tran= tf.expand_dims(denominator_stat_tran,2)
    denominator_stat_tran = tf.tile(denominator_stat_tran, [1,1,HS])
    result_state_trans = tf.divide(numerator_stat_tran,denominator_stat_tran)

    #PRIOR
    aux = tf.stack(aux_list_prior,0)
    summed_prior = tf.reduce_sum(aux,[3,2,0])#-------------------------------------------------------------CONTROLLARE L'INDICE I,J
    print("prior",aux)
    print("prior",summed_prior)
    n_l_list= tf.expand_dims(n_l_list,1)
    n_l_list = tf.tile(n_l_list, [1,HS])
    result_prior = tf.divide(summed_prior,n_l_list)

    return result_prior,result_sp,result_state_trans

############################################################################################################àà

HS =3
print("leggo il dataset")
data = dataset_parser()
EE_list = []
ee_list = []

for i in range(0,len(data)):
    EE = tf.placeholder(shape=[data[i].size,HS,HS], dtype=tf.float64)
    EE_list.append(EE)

    ee = np.zeros([data[i].size,HS,HS])
    w=0
    for i in range(1,data[i].size):
        for j in range(0,HS):
            for k in range(0,HS):
                ee[i,j,k]=w
                w=w+1
    ee_list.append(ee)

#for i in range(0,1000):
start = datetime.now()

result_prior,result_sp,result_state_trans = M_step(EE_list,data)

print("TEMPO TOTALE : ", (datetime.now() - start).total_seconds())

with tf.Session() as sess:

   sess.run(tf.global_variables_initializer(),)
   print(sess.run([result_prior,result_sp,result_state_trans], {
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
       EE_list[499]: ee_list[499]
   }))
