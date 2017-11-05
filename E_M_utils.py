import numpy as np
import tensorflow as tf
from tre_simple import *
from parser import *
#import pylab as pl

#np.set_printoptions(threshold=np.nan)

#hidden_state = 10
N_SYMBOLS = 367
MAX_CHILD = 32
CLASSI = 11

def modello(data_set,epoche,hidden_state):

    pi_l=  [[],[],[],[],[],[],[],[],[],[],[]]
    sp_p_l=[[],[],[],[],[],[],[],[],[],[],[]]
    bi_l=  [[],[],[],[],[],[],[],[],[],[],[]]
    A_l=   [[],[],[],[],[],[],[],[],[],[],[]]
    for i in range(0,CLASSI):
        #print("-------------------------------------------i =",i)
        pi_l[i],sp_p_l[i],A_l[i],bi_l[i] = training(data_set[i],epoche,hidden_state)

    return pi_l,sp_p_l,A_l,bi_l



def testing(data_test,pi_l,sp_p_l,A_l,bi_l,hidden_state):


    #class_result = np.zeros(len(data_test))
    class_result = tf.zeros([len(data_test)],dtype=tf.float64)

    #confusion_matrix = np.zeros((CLASSI,CLASSI))
    confusion_matrix = tf.zeros([CLASSI,CLASSI],dtype=tf.float64)

    giusti=0
    errati=0
    for j in range(0,len(data_test)):
        #print(j)
        like_max = -9999999999999999999

        for i in range(0,CLASSI):

            with tf.Session() as sess:

                var_EE, var_E = Reversed_Upward_Downward(sp_p_l[i], A_l[i], bi_l[i], pi_l[i], data_test[j],hidden_state)
                var_EE,var_E = sess.run([var_EE,var_E])

                sess.close
            tf.reset_default_graph() 

            with tf.Session() as sess:
                    
                like = log_likelihood_test(pi_l[i],sp_p_l[i],A_l[i],bi_l[i],var_EE,var_E,data_test[j],hidden_state)
                like = sess.run(like)

                sess.close
            tf.reset_default_graph() 


            if(like>like_max):
                class_result[j]=i+1
                like_max=like

        #print("reale ",data_test[j].classe,"    predetto",class_result[j])

        confusion_matrix[int(data_test[j].classe)-1][int(class_result[j])-1]= confusion_matrix[int(data_test[j].classe)-1][int(class_result[j])-1] +1
        
        if(  int(data_test[j].classe)    ==   int(class_result[j])   ):

            giusti=giusti+1

        else: 

            errati = errati +1


    rate =( (giusti) / (giusti+errati) ) * 100 
    print("giusti ",giusti)
    print("errati ",errati)
    print("rate   ",    rate )
    print("confusion_matrix\n",confusion_matrix)
    #np.savetxt('classi_risultato.out', class_result) 
    #np.savetxt('rate.out', rate) 

    return rate 



    
def training(data_set,epoche,hidden_state,pi=None,sp_p=None,A=None,bi=None):

    n=0
    s_1=[]
    s_2=[]
    s_3=[]
    s_4=[]
    like_list =[]

    #hidden_state da 2 a 20 non di piu, va calcolato l'algoritmo per i vari valori, che fanno cambiare il tutto di molto IMPORTANTE



    #nel caso non vengano passati dei valori iniziali ai parametri essi venono inizializati random
    if pi is None:
        pi = random_sum_one2(0, hidden_state, MAX_CHILD)
    if sp_p is None:
        sp_p = random_sum_one1(MAX_CHILD)
    if A is None:
        A = random_sum_one3(0, hidden_state, hidden_state, MAX_CHILD)
    if bi is None:
        bi = random_sum_one2(1, hidden_state, N_SYMBOLS)

    #per il numero delle epoco eseguo l'E-M

    for i in range(0, epoche):
        #print("EPOCA: ",i)

        var_EE_list = []
        var_E_list = []

        #eseguo E-STEP per ogni albero nel dataset


        for j in range(0,len(data_set)):
            #with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
            with tf.Session() as sess:

                var_EE, var_E = Reversed_Upward_Downward(sp_p, A, bi, pi, data_set[j],hidden_state)

                var_EE,var_E = sess.run([var_EE,var_E])


                var_EE_list.append(var_EE)
                var_E_list.append(var_E)
                sess.close

            tf.reset_default_graph()

        #with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
        with tf.Session() as sess:

            new_pi, new_sp_p, new_A, new_bi = M_step(var_EE_list,var_E_list,data_set,hidden_state)

            new_pi,new_sp_p,new_A,new_bi = sess.run([new_pi,new_sp_p,new_A,new_bi])

            sess.close
        tf.reset_default_graph()

        pi = new_pi
        sp_p = new_sp_p
        A = new_A
        bi = new_bi
    tf.reset_default_graph()


    return pi,sp_p,A,bi



def likelihood_test(data_set,epoche,hidden_state,pi=None,sp_p=None,A=None,bi=None):

    n=0
    s_1=[]
    s_2=[]
    s_3=[]
    s_4=[]
    like_list =[]

    #hidden_state da 2 a 20 non di piu, va calcolato l'algoritmo per i vari valori, che fanno cambiare il tutto di molto IMPORTANTE



    #nel caso non vengano passati dei valori iniziali ai parametri essi venono inizializati random
    if pi is None:
        pi = random_sum_one2(0, hidden_state, MAX_CHILD)
    if sp_p is None:
        sp_p = random_sum_one1(MAX_CHILD)
    if A is None:
        A = random_sum_one3(0, hidden_state, hidden_state, MAX_CHILD)
    if bi is None:
        bi = random_sum_one2(1, hidden_state, N_SYMBOLS)

    #per il numero delle epoco eseguo l'E-M

    for i in range(0, epoche):
        print("EPOCA: ",i)

        var_EE_list = []
        var_E_list = []

        #eseguo E-STEP per ogni albero nel dataset


        for j in range(0,len(data_set)):
            #with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
            with tf.Session() as sess:

                var_EE, var_E = Reversed_Upward_Downward(sp_p, A, bi, pi, data_set[j],hidden_state)

                var_EE,var_E = sess.run([var_EE,var_E])


                var_EE_list.append(var_EE)
                var_E_list.append(var_E)
                sess.close

            tf.reset_default_graph()

        #with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
        with tf.Session() as sess:

            new_pi, new_sp_p, new_A, new_bi = M_step(var_EE_list,var_E_list,data_set,hidden_state)

            new_pi,new_sp_p,new_A,new_bi = sess.run([new_pi,new_sp_p,new_A,new_bi])

            sess.close
        tf.reset_default_graph()

        pi = new_pi
        sp_p = new_sp_p
        A = new_A
        bi = new_bi

       # with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
        with tf.Session() as sess:

            s1, s2, s3, s4, tot = log_likelihood(pi,sp_p,A,bi,var_EE_list,var_E_list,data_set,hidden_state)

            tot ,s1, s2, s3 , s4 = sess.run([tot, s1 , s2 , s3 , s4])
            
            s_1.append(s1)
            s_2.append(s2)
            s_3.append(s3)
            s_4.append(s4)
            like_list.append(tot)

            sess.close
        tf.reset_default_graph()
    tf.reset_default_graph()


    #tf.reset_default_graph()

    #pl.plot(s_4,color='red')
    #pl.plot(s_3,color='blue')
    #pl.plot(s_2,color='orange')
    #pl.plot(s_1,color='green')
    #pl.plot(like_list)


    #np.savetxt('55like_list.out', like_list) 
    #np.savetxt('55s1.out', s_1) 
    #np.savetxt('55s2.out', s_2) 
    #np.savetxt('55s3.out', s_3) 
    #np.savetxt('55s4.out', s_4) 
    #pl.show()
    #pl.savefig('10.png')


    return pi,sp_p,A,bi


#||||||||||||||||||||||||||||||||||||||||||||||util||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||


def Reversed_Upward_Downward(sp_p, A, bi, pi, t,hidden_state):

    # upward parameters beta
    var_a_up_ward = tf.zeros([t.size, hidden_state],dtype=tf.float64)
    var_up_ward = tf.zeros([t.size, hidden_state],dtype=tf.float64)

    # pairwise smoothed posterior
    var_E = tf.zeros([t.size-1,hidden_state],dtype=tf.float64)
    var_EE = tf.zeros([t.size, hidden_state, hidden_state],dtype=tf.float64)


    var_in_prior = tf.fill([hidden_state, t.size], tf.cast(1/hidden_state, dtype=tf.float64) )



    #############################casi base ##############################################################################################

    var_up_ward,ris_17_t = compute_17(bi,pi,var_up_ward,t,hidden_state)

    var_in_prior = compute_internal_node_prior(var_in_prior, sp_p, A, t,hidden_state)

    var_a_up_ward = a_up_ward_foglie(var_a_up_ward,ris_17_t,A,var_in_prior,t,hidden_state)

    #up step
    for i in range(len(t.struct) - 2, -1, -1):

        var_up_ward,ris_19 = compute_19(A, bi, sp_p, var_a_up_ward, var_in_prior, var_up_ward, t,i,hidden_state)

        var_a_up_ward = compute_21(A,var_in_prior,var_a_up_ward,ris_19,i,t,hidden_state)

    var_E = set_base_case(var_up_ward,var_E,hidden_state)

    # down step
    for i in range(1, len(t.struct)  ):

        ris_24 = compute_24(sp_p, A, var_E, var_EE, var_up_ward, var_in_prior, var_a_up_ward, t, i ,hidden_state)

        var_EE = inglobe_ris_liv(ris_24, var_EE, t, i,hidden_state)

        var_E =  compute_25(ris_24, var_E, i, t,hidden_state)



    return (var_EE,var_E)

def compute_17(bi,pi,var_up_ward,t,hidden_state):

    label = []
    posizione = []
    for node in t.struct[-1]:
        label.append(node.label)
        posizione.append(node.pos -1)
    #prelevo da bi e pi i dati relativi alle foglie
    aux1 = tf.gather(bi, label, axis=1)
    aux2 = tf.gather(pi, posizione, axis=1)

    nume = tf.multiply(aux1, aux2)  # Element-wise multiplication
    den = tf.reduce_sum(nume,[0])
    # righe e colonne con lo stesso indice
    ris_17_t = tf.divide(nume, den)

    ris_17_t = tf.transpose(ris_17_t, perm=[1, 0])

    head = tf.slice(var_up_ward, [0, 0], [t.struct[-1][0].name, hidden_state])
    var_up_ward = tf.concat([head, ris_17_t], 0)

    return var_up_ward, ris_17_t

def compute_internal_node_prior(var_in_prior,sp_p,A,t,hidden_state):

    aux1 = tf.multiply(sp_p, A)  # broadcast implicito


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

        aux2 = tf.gather(var_in_prior, nomi_figli, axis=1)

        aux2 = tf.transpose(aux2, perm=[1, 0, 2])

        aux2 = tf.expand_dims(aux2, 1)

        aux2 = tf.tile(aux2, [1, hidden_state, 1, 1])

        #  di figli di un nodo  in un unica matrice hidden_state*L*(numero di nodi del livello)
        # qui moltiplicazione
        # questa e una serie di matrici, tante quanti sono i nodi del livello esaminati

        aux3 = tf.multiply(aux1, aux2)
        s = tf.reduce_sum(aux3, [2, 3])
        s = tf.transpose(s)

        # prelevo i valori iniziali e quelli finali che non devono essere aggiornati in questo ciclo
        head = tf.slice(var_in_prior, [0, 0], [hidden_state,t.struct[i][-1].name + 1 - int((s.shape[1]))])  # ricorda che questa deriva da quella sopra
        # [hidden_state, (t.size  -(t.size - t.struct[i][-1].name - 1) - int((s.shape[1])) )])

        tail = tf.slice(var_in_prior, [0, t.struct[i][-1].name + 1],
                        [hidden_state,
                         t.size - t.struct[i][-1].name - 1])  # potrei farlo anche con un constant

        var_in_prior = tf.concat([head, s, tail], 1)  # aggiorno i nuovi valore trovati
    return var_in_prior

def a_up_ward_foglie(var_a_up_ward,ris_17_t,A,var_in_prior,t,hidden_state):

    padri = []
    posizione = []
    for node in t.struct[-1]:
        padri.append(node.father.name)
        posizione.append(node.pos-1) 

    slice_A = tf.gather(A, posizione, axis=2)
    slice_in_prior = tf.gather(var_in_prior, padri, axis=1)

    up_war_foglie = tf.expand_dims(ris_17_t,2)
    up_war_foglie = tf.tile(up_war_foglie, [1, 1, hidden_state])
    up_war_foglie = tf.transpose(up_war_foglie,[2,1,0])         #DDDD------------------------------------------------------ ji
    #up_war_foglie = tf.transpose(up_war_foglie,[1,2,0])         #DDDD------------------------------------------------------ ji

    numerator = tf.multiply(slice_A, up_war_foglie)
    numerator = tf.reduce_sum(numerator,[1])
    a_up_war_foglie = tf.divide(numerator,slice_in_prior)
    a_up_war_foglie = tf.transpose(a_up_war_foglie,[1,0])

    head = tf.slice(var_a_up_ward, [0, 0], [t.struct[-1][0].name, hidden_state])
    var_a_up_ward = tf.concat([head, a_up_war_foglie], 0)
    return var_a_up_ward

def compute_19(A, bi, sp_p, a_up_ward, var_in_prior, var_up_ward, t, i,hidden_state):



    # mi calcolo il numeratore della
    posizione = []
    nomi = []
    for node in t.struct[i]:
        posizione.append([])
        nomi.append([])
        for child in node.children:
            posizione[-1].append(child.pos-1)
            nomi[-1].append(child.name)
        for k in range(len(nomi[-1]),MAX_CHILD):
            posizione[-1].append(0)
            nomi[-1].append(0) 

    slice_A = tf.gather(A, posizione, axis=2)

    slice_A = tf.transpose(slice_A,[2,0,1,3])         #DDDD------------------------------------------------------ ji
    #slice_A = tf.transpose(slice_A,[2,1,0,3])         #DDDD------------------------------------------------------ ji

    slice_var_up_war_provvisoria = tf.gather(var_up_ward, nomi, axis=0)
    slice_var_up_war_provvisoria = tf.expand_dims(slice_var_up_war_provvisoria,2)
    slice_var_up_war_provvisoria = tf.tile(slice_var_up_war_provvisoria, [1, 1, hidden_state, 1])
    slice_var_up_war_provvisoria = tf.transpose(slice_var_up_war_provvisoria,[0,3,2,1])         #DDDD------------------------------------------------------ ji
    #slice_var_up_war_provvisoria = tf.transpose(slice_var_up_war_provvisoria,[0,2,3,1])         #DDDD------------------------------------------------------ ji


    numerator = tf.multiply(slice_A, slice_var_up_war_provvisoria)
    numerator_21 = tf.reduce_sum(numerator,[1])



    sli_var_in_prior = tf.ones([hidden_state, 0], tf.float64)

    labels = []
    nomi = []
    figli = []
    for node in t.struct[i]:
        labels.append(node.label)
        nomi.append(node.name)
        figli.append([])
        for child in node.children:
           figli[-1].append(child.name)
        for j in range(len(figli[-1]), MAX_CHILD):#MMM qui puoi farlo fino a max lunghezza invehe che
            figli[-1].append(0)
    sli_ph_bi = tf.gather(bi, labels, axis=1)

    aux_sp = tf.expand_dims(sp_p,0)
    aux_sp = tf.expand_dims(aux_sp,0)
    aux_sp = tf.tile(aux_sp,[len(t.struct[i]),hidden_state,1])

    second_term_numerator_19_inter = tf.multiply(aux_sp, numerator_21)
    second_term_numerator_19 = tf.reduce_sum(second_term_numerator_19_inter,[2])
    second_term_numerator_19 = tf.transpose(second_term_numerator_19,[1,0])
    full_numerator_19 = tf.multiply(sli_ph_bi, second_term_numerator_19)


    sli_ph_bi = tf.expand_dims(sli_ph_bi, 2)
    sli_ph_bi = tf.tile(sli_ph_bi, [1, 1, MAX_CHILD])
    second_term_numerator_19_inter = tf.transpose(second_term_numerator_19_inter,[1,0,2])

    den_to_sum = tf.multiply(sli_ph_bi,second_term_numerator_19_inter)
    full_denominator19 = tf.reduce_sum(den_to_sum,[2,0])

    ris_19 = tf.divide(full_numerator_19,full_denominator19)
    ris_19 = tf.transpose(ris_19,[1,0])
    head = tf.slice(var_up_ward, [0, 0], [t.struct[i][-1].name + 1 - int((ris_19.shape[0])), hidden_state])
    tail = tf.slice(var_up_ward, [t.struct[i][-1].name +1 , 0], [t.size - t.struct[i][-1].name - 1, hidden_state])



    var_up_ward = tf.concat([head, ris_19, tail], 0)

    return var_up_ward,ris_19

def compute_21(A,var_in_prior,var_a_up_ward,ris_19,i,t,hidden_state):

    if (i != 0):
        padri = []
        posizione = []
        for node in t.struct[i]:
            padri.append(node.father.name)
            posizione.append(node.pos-1) 

        slice_A = tf.gather(A, posizione, axis=2)
        slice_in_prior = tf.gather(var_in_prior, padri, axis=1)

        up_war_foglie = tf.expand_dims(ris_19,2)
        up_war_foglie = tf.tile(up_war_foglie, [1, 1, hidden_state])
        up_war_foglie = tf.transpose(up_war_foglie,[2,1,0])         #DDDD------------------------------------------------------ ji

        numerator = tf.multiply(slice_A, up_war_foglie)
        numerator = tf.reduce_sum(numerator,[1])
        a_up_war_foglie = tf.divide(numerator,slice_in_prior)
        a_up_war_foglie = tf.transpose(a_up_war_foglie,[1,0])

        head = tf.slice(var_a_up_ward, [0, 0], [t.struct[i][0].name, hidden_state])
        tail = tf.slice(var_a_up_ward, [t.struct[i][-1].name +1 , 0], [t.size - t.struct[i][-1].name - 1, hidden_state])

        var_a_up_ward = tf.concat([head, a_up_war_foglie,tail], 0)


    return var_a_up_ward

def set_base_case(var_up_ward,var_E,hidden_state):

    base_case = tf.slice(var_up_ward, [0, 0], [1, hidden_state])
    var_E = tf.concat([base_case, var_E], 0)

    return var_E

def compute_24(sp_p, A, var_E, var_EE, var_up_ward, var_in_prior, var_a_up_ward, t, i,hidden_state ):

    padri = []
    posizione = []
    nomi_nodi = []
    fratelli = []
    fratelli_pos = []
    max_fratelli=0
    for node in t.struct[i]:
        nomi_nodi.append(node.name)
        padri.append(node.father.name)
        posizione.append(node.pos-1)
        fratelli.append([])
        fratelli_pos.append([])
        k=0
        for child in node.father.children:
            fratelli[-1].append(child.name)
            fratelli_pos[-1].append(child.pos-1)
            k = k+1
        if (k>= max_fratelli):
            max_fratelli=k

    for lista_frat in fratelli_pos:
        while(len(lista_frat)<max_fratelli):
            lista_frat.append(0)
    for lista_frat in fratelli:
        while(len(lista_frat)<max_fratelli):
            lista_frat.append(0)         
            

    sli_E = tf.gather(var_E, padri)
    sli_up_ward = tf.gather(var_up_ward, nomi_nodi)
    sli_sp_p_aux = tf.gather(sp_p, posizione)
    sli_A = tf.gather(A, posizione, axis=2)
    sli_A = tf.transpose(sli_A, perm=[2, 0, 1])   #DDD   --------------------ij
    #per il den
    sli_in_prior = tf.gather(var_in_prior, padri, axis=1)
    sli_var_a_up_ward = tf.gather(var_a_up_ward, fratelli)             # qui non e padri ma e FRATELLI del nodo in questione
    sli_sp_pos = tf.gather(sp_p,fratelli_pos)
    # per il numeratore
    sli_E = tf.expand_dims(sli_E, 1)
    sli_E = tf.tile(sli_E, [ 1,hidden_state, 1])


    sli_up_ward = tf.expand_dims(sli_up_ward, 1)
    sli_up_ward = tf.tile(sli_up_ward, [1, hidden_state, 1])

    sli_sp_p = tf.expand_dims(sli_sp_p_aux, 1)
    sli_sp_p = tf.expand_dims(sli_sp_p, 1)
    sli_sp_p = tf.tile(sli_sp_p, [1, hidden_state, hidden_state])


    numerator = tf.multiply(sli_E, sli_up_ward)
    numerator = tf.multiply(numerator, sli_sp_p)
    numerator = tf.multiply(numerator, sli_A)


    # per il denominatore

    sli_sp_pos = tf.expand_dims(sli_sp_pos, 2)
    sli_sp_pos = tf.tile(sli_sp_pos, [1,1,hidden_state])


    to_sum = tf.multiply(sli_sp_pos, sli_var_a_up_ward)

    added = tf.reduce_sum(to_sum, [1])  # sommo nella dim 2

    sli_in_prior = tf.transpose(sli_in_prior, perm=[1, 0])

    denominator = tf.multiply(sli_in_prior, added)
    denominator = tf.expand_dims(denominator, 1)
    denominator = tf.tile(denominator, [1, hidden_state, 1])

    denominator = tf.add(denominator, tf.constant(1/10000000, dtype=tf.float64,shape=denominator.shape))
    numerator = tf.add(numerator, tf.constant(1/10000000, dtype=tf.float64,shape=numerator.shape))

    ris_24 = tf.divide(numerator, denominator)

    #uniformarel la somma in moso che faccio uno su j+i (tutto)

    uniform = tf.reduce_sum(ris_24, [1,2])
    uniform = tf.expand_dims(uniform, 1)
    uniform = tf.expand_dims(uniform, 1)
    uniform = tf.tile(uniform, [1, hidden_state,hidden_state])
    ris_24 = tf.divide(ris_24, uniform)

    return ris_24
#funzione che
def inglobe_ris_liv(ris_24, var_EE, t, i,hidden_state):

    head = tf.slice(var_EE, [0, 0, 0],
                    [t.struct[i][-1].name + 1 - int((ris_24.shape[0])), hidden_state,
                     hidden_state])
    tail = tf.slice(var_EE, [t.struct[i][-1].name + 1, 0, 0],
                    [t.size - t.struct[i][-1].name - 1, hidden_state, hidden_state])

    ris_24 = tf.add(ris_24, tf.constant(1/10000000, dtype=tf.float64,shape=ris_24.shape))

    var_EE = tf.concat([head, ris_24, tail], 0)

    return var_EE

def compute_25(ris_24, var_E, i, t,hidden_state):

    ris_25 = tf.reduce_sum(ris_24, [2])


    uniform = tf.reduce_sum(ris_25, [1])
    uniform = tf.expand_dims(uniform, 1)
    uniform = tf.tile(uniform, [1, hidden_state])
    ris_25 = tf.divide(ris_25, uniform)


    head = tf.slice(var_E, [0, 0], [t.struct[i][-1].name + 1 - int((ris_24.shape[0])),
                                    hidden_state])  # _________________-da controllare la dim giusta in shape
    tail = tf.slice(var_E, [t.struct[i][-1].name + 1, 0],
                    [t.size - t.struct[i][-1].name - 1, hidden_state])

    var_E = tf.concat([head, ris_25, tail], 0)

    return var_E



#funzione ausiliaria per inizializzare in modo casuale tensori di 1 dimenzione
def random_sum_one1(shape1):

    rand = np.random.uniform( 0, 1, [shape1])
    sum = np.sum(rand, 0)

    rand_sum_one = np.divide(rand, sum)

    return rand_sum_one
#funzione ausiliaria per inizializzare in modo casuale tensori di 2 dimenzioni dati l'asse di
def random_sum_one2(axe,shape1,shape2):

    rand = np.random.uniform( 0, 1, [shape1, shape2])
    sum = np.sum(rand, axe)

    #nel caso l'asse non e lo zero lo espando duplico cosi da poter dividere la matrice random per esso
    if axe == 1:
        sum = np.expand_dims(sum, 1)
        sum = np.tile(sum, [1, shape2])

    rand_sum_one = np.divide(rand, sum)

    return rand_sum_one
#funzione ausiliaria per inizializzare in modo casuale tensori di 3 dimenzioni
def random_sum_one3(axe,shape1,shape2,shape3=None):

    rand = np.random.uniform(0, 1, [shape1, shape2, shape3])
    sum = np.sum(rand, axe)
    sum = np.expand_dims(sum, axe)
    sum = np.tile(sum, [shape2,1,1])
    rand_sum_one = np.divide(rand, sum)

    return rand_sum_one


def M_step(var_EE_list,var_E_list,data_set,hidden_state):


    with tf.Session() as sess:

        lista_prior = []
        lista_n_in = []
        aux_list_prior = []
        aux_list_sp = []
        sum_N_I = 0
        max_l = -1
        num_tm_yu_list = []
        den_tm_yu_list = []

        # MULTINOMIAL
        # calcolo resized_tm_yu##
        #                                        DDDD   uesto puo essere fatta in maniera migliore e piu semplice andando a lavorare sugli indici
        for i in range(0, len(var_EE_list)):
            tm_yu = np.zeros([int(data_set[i].size), N_SYMBOLS])
            for level in data_set[i].struct:
                for node in level:
                    tm_yu[node.name, node.label] = 1

            e_resized = tf.expand_dims(var_E_list[i], 2)
            e_resized = tf.tile(e_resized, [1, 1, N_SYMBOLS])

            resized_tm_yu = tf.expand_dims(tm_yu, 2)  # qui posso portare fuori dal for per ottimizzare...
            resized_tm_yu = tf.tile(resized_tm_yu, [1, 1, hidden_state])
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

        numerator_tm_yu = tf.add(numerator_tm_yu, tf.constant(1/10000000, dtype=tf.float64,shape=numerator_tm_yu.shape))
        result_multinomial = tf.divide(numerator_tm_yu, denominator_tm_yu)

        # calcolo il numero totale di nodi nell L-esima posizione
        # e anche il numero massimo di nodi nella l-esima posizione, cosi da poter dimenzionare in maniera opportuna le dim
        for i in range(0, len(data_set)):
            #aggiungo una lista per ogni albero
            lista_prior.append([])
            lista_n_in.append([])
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
            # uniformo la lunghezza cosi da non rendere il tensore sparso per la futura gather
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

        summed_sp_denominator = tf.reduce_sum(summed_sp,[0])
        summed_sp_denominator = tf.expand_dims(summed_sp_denominator, 0)
        summed_sp_denominator = tf.tile(summed_sp_denominator, [ MAX_CHILD])

        summed_sp2 = tf.reduce_sum(aux, [4, 3])
        summed_sp = tf.add(summed_sp, tf.constant(1/10000000, dtype=tf.float64,shape=summed_sp.shape))
        result_sp = tf.divide(summed_sp, summed_sp_denominator)

        # STATE TRANSICTION (A)
        numerator_stat_tran = tf.reduce_sum(aux, [2,0])

        denominator_stat_tran = tf.reduce_sum(numerator_stat_tran, [2])
        denominator_stat_tran = tf.expand_dims(denominator_stat_tran, 2)
        denominator_stat_tran = tf.tile(denominator_stat_tran, [1, 1, hidden_state])

        numerator_stat_tran = tf.add(numerator_stat_tran, tf.constant(1/10000000, dtype=tf.float64,shape=numerator_stat_tran.shape))
        result_state_trans = tf.divide(numerator_stat_tran, denominator_stat_tran)
        result_state_trans = tf.transpose(result_state_trans, [2, 1, 0])
        result_state_trans = tf.where(tf.is_inf(result_state_trans),tf.constant(1/hidden_state, dtype=tf.float64,shape=result_state_trans.shape), result_state_trans)
        result_state_trans = tf.where(tf.is_nan(result_state_trans), tf.constant(1/hidden_state, dtype=tf.float64,shape=result_state_trans.shape), result_state_trans)

        # PRIOR
        aux = tf.stack(aux_list_prior, 0)

        summed_prior = tf.reduce_sum(aux, [3, 2, 0])#DDD

        denominatore_summed_prior = tf.reduce_sum(summed_prior,[1])
        denominatore_summed_prior = tf.expand_dims(denominatore_summed_prior, 1)
        denominatore_summed_prior = tf.tile(denominatore_summed_prior, [1, hidden_state])
        summed_prior = tf.add(summed_prior, tf.constant(1/10000000, dtype=tf.float64,shape=summed_prior.shape))
        
        result_prior = tf.divide(summed_prior, denominatore_summed_prior)
        result_prior = tf.where(tf.is_inf(result_prior), tf.constant(1/hidden_state, dtype=tf.float64,shape=result_prior.shape), result_prior)      
        result_prior = tf.where(tf.is_nan(result_prior), tf.constant(1/hidden_state, dtype=tf.float64,shape=result_prior.shape), result_prior)
        result_prior = tf.transpose(result_prior, [1, 0])

        #DDD devo far si che la somma suglii ia uno?
        pi,sp_p,bi,A = sess.run([result_prior,result_sp,result_multinomial,result_state_trans])

        sess.close

    return result_prior,result_sp,result_state_trans,result_multinomial

def log_likelihood(pi,sp_p,A,bi,var_EE_list,var_E_list,data_set,hidden_state):


    tot = 0
    s1=0
    s2=0
    s3=0
    s4=0
    for i in range(0, len(data_set)):

        # prelevo i nodi interni e foglia
        leaf_node = tf.slice(var_E_list[i], [data_set[i].struct[-1][0].name, 0],
                             [data_set[i].size - data_set[i].struct[-1][0].name, hidden_state])

        # prima sommatoria
        # salvo e prelevo la lista dell'indicatore posizionale di ogni nodo foglia
        posizione_foglie = []
        for node in data_set[i].struct[-1]:
            posizione_foglie.append(node.pos-1)
        log_pi = tf.gather(pi, posizione_foglie, axis=1)
        log_pi = tf.log(log_pi)
        log_pi = tf.transpose(log_pi, [1, 0])
        log_pi = tf.where(tf.is_inf(log_pi), tf.zeros_like(log_pi), log_pi)
        #log_pi = tf.where(tf.less(log_pi,tf.constant(1/TO_ZERO, dtype=tf.float64,shape=log_pi.shape)), tf.zeros(dtype=tf.float64,shape=log_pi.shape), log_pi)

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
        log_bi = tf.where(tf.is_inf(log_bi), tf.zeros_like(log_bi), log_bi)

        log_bi = tf.cast(log_bi, tf.float64)
        seconda_somm = tf.multiply(log_bi, var_E_list[i])
        seconda_somm = tf.reduce_sum(seconda_somm, [0, 1])

        # terza sommatoria

        posizione_nodi_interni = []
        for j in range(0, data_set[i].struct[-1][0].name):
            posizione_nodi_interni.append([])

        internal_node_ee = tf.slice(var_EE_list[i], [0, 0, 0],
                                    [data_set[i].struct[-1][0].name, hidden_state, hidden_state])
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
        log_A = tf.where(tf.is_inf(log_A), tf.zeros_like(log_A), log_A)
        log_A = tf.cast(log_A, tf.float64)
        quarta_somm = tf.multiply(pqqsy,
                                  log_A)  # ________________________________________________________indicie ij da controllare
        quarta_somm = tf.reduce_sum(quarta_somm, [1, 2, 3, 0])

        tot = tot + prima_somm + seconda_somm + terza_somm + quarta_somm
        s1= s1+prima_somm
        s2=s2+seconda_somm
        s3=s3+terza_somm
        s4=s4+quarta_somm



    return s1,s2,s3,s4, tot

def log_likelihood_test(pi,sp_p,A,bi,var_EE_list,var_E_list,t,hidden_state):




    # prelevo i nodi interni e foglia
    leaf_node = tf.slice(var_E_list, [t.struct[-1][0].name, 0],
                         [t.size - t.struct[-1][0].name, hidden_state])

    # prima sommatoria
    # salvo e prelevo la lista dell'indicatore posizionale di ogni nodo foglia
    posizione_foglie = []
    for node in t.struct[-1]:
        posizione_foglie.append(node.pos-1)
    log_pi = tf.gather(pi, posizione_foglie, axis=1)
    log_pi = tf.log(log_pi)
    log_pi = tf.transpose(log_pi, [1, 0])
    log_pi = tf.where(tf.is_inf(log_pi), tf.zeros_like(log_pi), log_pi)
    #log_pi = tf.where(tf.less(log_pi,tf.constant(1/TO_ZERO, dtype=tf.float64,shape=log_pi.shape)), tf.zeros(dtype=tf.float64,shape=log_pi.shape), log_pi)

    log_pi = tf.cast(log_pi, tf.float64)

    prima_somm = tf.multiply(log_pi, leaf_node)
    prima_somm = tf.reduce_sum(prima_somm, [0, 1])

    # seconda sommatoria
    label_nodi = []
    for level in t.struct:
        for node in level:
            label_nodi.append(node.label)
    log_bi = tf.gather(bi, label_nodi, axis=1)
    log_bi = tf.log(log_bi)
    log_bi = tf.transpose(log_bi, [1, 0])
    log_bi = tf.where(tf.is_inf(log_bi), tf.zeros_like(log_bi), log_bi)

    log_bi = tf.cast(log_bi, tf.float64)
    seconda_somm = tf.multiply(log_bi, var_E_list)
    seconda_somm = tf.reduce_sum(seconda_somm, [0, 1])

    # terza sommatoria

    posizione_nodi_interni = []
    for j in range(0, t.struct[-1][0].name):
        posizione_nodi_interni.append([])

    internal_node_ee = tf.slice(var_EE_list, [0, 0, 0],
                                [t.struct[-1][0].name, hidden_state, hidden_state])
    internal_node_ee = tf.reduce_sum(internal_node_ee, [2, 1])

    for level in t.struct[:-1]:
        for node in level:
            for child in node.children:
                posizione_nodi_interni[node.name].append(child.name)

    for in_list in posizione_nodi_interni:
        start = len(in_list)
        for k in range(start, MAX_CHILD):
            in_list.append(0)

    ee_sum_c_c = tf.reduce_sum(var_EE_list, [2, 1])

    psul = tf.gather(ee_sum_c_c, posizione_nodi_interni)

    log_sp_p = tf.log(sp_p)

    log_sp_p = tf.where(tf.is_inf(log_sp_p), tf.zeros_like(log_sp_p), log_sp_p)

    log_sp_p = tf.cast(log_sp_p, tf.float64)
    terza_somm = tf.multiply(psul, log_sp_p)
    terza_somm = tf.reduce_sum(terza_somm, [0, 1])

    # QUARTA SOMMATORIA
    pqqsy = tf.gather(var_EE_list, posizione_nodi_interni)

    log_A = tf.transpose(A, [2, 1, 0])
    log_A = tf.log(log_A)
    log_A = tf.where(tf.is_inf(log_A), tf.zeros_like(log_A), log_A)
    log_A = tf.cast(log_A, tf.float64)
    quarta_somm = tf.multiply(pqqsy,
                              log_A)  # ________________________________________________________indicie ij da controllare
    quarta_somm = tf.reduce_sum(quarta_somm, [1, 2, 3, 0])

    tot =  prima_somm + seconda_somm + terza_somm + quarta_somm


    return tot

    
def divide_tre_validation (dataset):

    d_dataset = [[[],[]],[[],[]],[[],[]]]

    for i in range(0,11):
        split_size = len(dataset[i])//3

    #A
        #traning set
        d_dataset[0][0].append([])
        d_dataset[0][0][i]=dataset[i][:split_size*2]
        #validation set
        d_dataset[0][1] += dataset[i][split_size*2:]

        #B
        d_dataset[1][0].append([])
        d_dataset[1][0][i]=dataset[i][:split_size] + dataset[i][split_size*2:]
        #validation set
        d_dataset[1][1] += dataset[i][split_size:split_size*2]

        #C
        d_dataset[2][0].append([])
        d_dataset[2][0][i]=dataset[i][split_size:]
        #validation set
        d_dataset[2][1] += dataset[i][:split_size]

    return d_dataset
























