import tensorflow as tf
from tre_simple import *
from parser import *
#import pylab as pl

np.set_printoptions(threshold=np.nan)
N_HIDDEN_STATES = 6
N_SYMBOLS = 367
MAX_CHILD = 32
TO_ZERO =1000000000000000000

s_1=[]
s_2=[]
s_3=[]
s_4=[]


epoche = 75

#||||||||||||||||||||||||||||||||||||||||||||||||||||||||general||||||||||||||||||||||||||||||||||||||||||||||||||||||||
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




#|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||start||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
print("leggo il dataset")
data_set = dataset_parser()

#print("dataset lung  ",len(data_set))
n=0

#N_HIDDEN_STATES da 2 a 20 non di piu, va calcolato l'algoritmo per i vari valori, che fanno cambiare il tutto di molto IMPORTANTE


scope_tree = "scope_n0"
scope_epoca = "epoca_n0"
#nel caso non vengano passati dei valori iniziali ai parametri essi venono inizializati random
pi = random_sum_one2(0, N_HIDDEN_STATES, MAX_CHILD)
sp_p = random_sum_one1(MAX_CHILD)
A = random_sum_one3(0, N_HIDDEN_STATES, N_HIDDEN_STATES, MAX_CHILD)
bi = random_sum_one2(1, N_HIDDEN_STATES, N_SYMBOLS)
like_list =[]


inizializzazione =True
#per il numero delle epoco eseguo l'E-M
for z in range(0, epoche):

    print("-----EPOCA: ",z)
    #scope_epoca = scope_epoca[:-len(str(i - 1))] + str(i)
    #with tf.variable_scope(scope_epoca):

    #eseguo E-STEP per ogni albero nel dataset
    var_EE_list = []
    var_E_list = []
    test1=[]
    test2=[]
    test3=[]
    scope_tree = "scopen0"
   # print(" E-step  ")


    for zzzz in range(0,len(data_set)):

        with tf.Session() as sess:
            #scope_tree=scope_tree[:-len(str(j-1))]+str(j)
            #print("scope_tree------------------------__>>>>>>>>>>>>>>",scope_tree)
            # |||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||Reversed_Upward_Downward||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
            #print("         albero:",j,end=" ")
            t= data_set[zzzz]
            # upward parameters beta
            up_ward = np.zeros((t.size, N_HIDDEN_STATES))
            a_up_ward = np.zeros((t.size, N_HIDDEN_STATES))

            # internal node prior
            in_prior = np.ones((N_HIDDEN_STATES, t.size))
            #in_prior = np.zeros((N_HIDDEN_STATES, t.size))

            # pairwise smoothed posterior
            E = np.zeros((t.size-1,N_HIDDEN_STATES))
            EE = np.zeros((t.size, N_HIDDEN_STATES, N_HIDDEN_STATES))

            #for ii in range(0, N_HIDDEN_STATES):
            #   for jj in range(0, N_HIDDEN_STATES):
            #      EE[0,ii,jj]=1/N_HIDDEN_STATES

            for ii in range(0, N_HIDDEN_STATES):
                #for jj in range(t.size - len(t.struct[-1]), t.size):
                for jj in range(0,t.size):
                    in_prior[ii, jj] = 1/N_HIDDEN_STATES

            var_in_prior = tf.constant(in_prior, dtype=tf.float64)

            var_a_up_ward = tf.constant(a_up_ward, dtype=tf.float64)
            var_up_ward = tf.constant(up_ward, dtype=tf.float64)

            var_E = tf.constant(E, dtype=tf.float64)
            var_EE = tf.constant(EE, dtype=tf.float64)


            # |||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||compute_17||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
            label = []
            posizione = []
            for node in t.struct[-1]:
                label.append(node.label)
                posizione.append(node.pos -1)
            #prelevo da bi e pi i dati relativi alle foglie
            aux1 = tf.gather(bi, label, axis=1)
            aux2 = tf.gather(pi, posizione, axis=1)

            nume = tf.multiply(aux1, aux2)  # Element-wise multiplication
            #den = tf.einsum('ij,ji->i', tf.transpose(aux1), aux2)  # Einstein summation per moltiplicazione di
            den = tf.reduce_sum(nume,[0])
            # righe e colonne con lo stesso indice
            ris_17_t = tf.divide(nume, den)

            test_aux1 = aux1
            test_aux2 = aux2
            ris_17_t = tf.transpose(ris_17_t, perm=[1, 0])

            #ris_17_t = tf.where(tf.is_nan(ris_17_t), tf.zeros_like(ris_17_t),ris_17_t)

            head = tf.slice(var_up_ward, [0, 0], [t.struct[-1][0].name, N_HIDDEN_STATES])
            var_up_ward = tf.concat([head, ris_17_t], 0)


            # |||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||compute_internal_node_prior||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
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
                #aux2 = tf.where(tf.is_zero(aux2), tf.zeros_like(aux2), aux2)

                aux2 = tf.transpose(aux2, perm=[1, 0, 2])

                aux2 = tf.expand_dims(aux2, 1)

                aux2 = tf.tile(aux2, [1, N_HIDDEN_STATES, 1, 1])

                #  di figli di un nodo  in un unica matrice N_HIDDEN_STATES*L*(numero di nodi del livello)
                # qui moltiplicazione
                # questa e una serie di matrici, tante quanti sono i nodi del livello esaminati

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



            # |||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||a_up_ward  sulle foglie ||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||

            padri = []
            posizione = []
            for node in t.struct[-1]:
                padri.append(node.father.name)
                posizione.append(node.pos-1) 

            slice_A = tf.gather(A, posizione, axis=2)
            slice_in_prior = tf.gather(var_in_prior, padri, axis=1)

            up_war_foglie = tf.expand_dims(ris_17_t,2)
            up_war_foglie = tf.tile(up_war_foglie, [1, 1, N_HIDDEN_STATES])
            up_war_foglie = tf.transpose(up_war_foglie,[2,1,0])         #DDDD------------------------------------------------------ ji
            #up_war_foglie = tf.transpose(up_war_foglie,[1,2,0])         #DDDD------------------------------------------------------ ji

            numerator = tf.multiply(slice_A, up_war_foglie)
            numerator = tf.reduce_sum(numerator,[1])
            a_up_war_foglie = tf.divide(numerator,slice_in_prior)
            a_up_war_foglie = tf.transpose(a_up_war_foglie,[1,0])

            head = tf.slice(var_a_up_ward, [0, 0], [t.struct[-1][0].name, N_HIDDEN_STATES])
            var_a_up_ward = tf.concat([head, a_up_war_foglie], 0)





            # |||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||up step||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
            # DDD non dovrebbe fare uno la somma dei 21? sulle i?
            
            # up step
            for i in range(len(t.struct) - 2,  -1  , -1):


                # |||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||compute_19||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
            
            # qui mi sto calcolando il numeratore dell 21 per uando lo devo utilizzare dentro la 19 
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
                slice_var_up_war_provvisoria = tf.tile(slice_var_up_war_provvisoria, [1, 1, N_HIDDEN_STATES, 1])
                slice_var_up_war_provvisoria = tf.transpose(slice_var_up_war_provvisoria,[0,3,2,1])         #DDDD------------------------------------------------------ ji
                #slice_var_up_war_provvisoria = tf.transpose(slice_var_up_war_provvisoria,[0,2,3,1])         #DDDD------------------------------------------------------ ji


                numerator = tf.multiply(slice_A, slice_var_up_war_provvisoria)
                numerator_21 = tf.reduce_sum(numerator,[1])



                sli_var_in_prior = tf.ones([N_HIDDEN_STATES, 0], tf.float64)

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
                aux_sp = tf.tile(aux_sp,[len(t.struct[i]),N_HIDDEN_STATES,1])

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
                head = tf.slice(var_up_ward, [0, 0], [t.struct[i][-1].name + 1 - int((ris_19.shape[0])), N_HIDDEN_STATES])
                tail = tf.slice(var_up_ward, [t.struct[i][-1].name +1 , 0], [t.size - t.struct[i][-1].name - 1, N_HIDDEN_STATES])
                # tail=tf.slice(var_up_ward, [t.struct[i][-1].name -1 , 0],   per far compilare



                var_up_ward = tf.concat([head, ris_19, tail], 0)


                #||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||  compute 21      
                
                if (i != 0):
                    padri = []
                    posizione = []
                    for node in t.struct[i]:
                        padri.append(node.father.name)
                        posizione.append(node.pos-1) 

                    slice_A = tf.gather(A, posizione, axis=2)
                    slice_in_prior = tf.gather(var_in_prior, padri, axis=1)

                    up_war_foglie = tf.expand_dims(ris_19,2)
                    up_war_foglie = tf.tile(up_war_foglie, [1, 1, N_HIDDEN_STATES])
                    up_war_foglie = tf.transpose(up_war_foglie,[2,1,0])         #DDDD------------------------------------------------------ ji
                    #up_war_foglie = tf.transpose(up_war_foglie,[1,2,0])         #DDDD------------------------------------------------------ ji
                    numerator = tf.multiply(slice_A, up_war_foglie)
                    numerator = tf.reduce_sum(numerator,[1])
                    a_up_war_foglie = tf.divide(numerator,slice_in_prior)
                    a_up_war_foglie = tf.transpose(a_up_war_foglie,[1,0])

                    head = tf.slice(var_a_up_ward, [0, 0], [t.struct[i][0].name, N_HIDDEN_STATES])
                    tail = tf.slice(var_a_up_ward, [t.struct[i][-1].name +1 , 0], [t.size - t.struct[i][-1].name - 1, N_HIDDEN_STATES])

                    var_a_up_ward = tf.concat([head, a_up_war_foglie,tail], 0)

                test = tf.reduce_sum(var_up_ward,[1])



                # |||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||compute_21||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||


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
                sli_E = tf.tile(sli_E, [ 1,N_HIDDEN_STATES, 1])


                sli_up_ward = tf.expand_dims(sli_up_ward, 1)
                sli_up_ward = tf.tile(sli_up_ward, [1, N_HIDDEN_STATES, 1])

                sli_sp_p = tf.expand_dims(sli_sp_p_aux, 1)
                sli_sp_p = tf.expand_dims(sli_sp_p, 1)
                sli_sp_p = tf.tile(sli_sp_p, [1, N_HIDDEN_STATES, N_HIDDEN_STATES])


                numerator = tf.multiply(sli_E, sli_up_ward)
                numerator = tf.multiply(numerator, sli_sp_p)
                numerator = tf.multiply(numerator, sli_A)


                test_sliE = sli_E
                test_sli_up_ward=sli_up_ward
                test_sli_sp_p=sli_sp_p
                test_sli_A=sli_A
                # per il denominatore

                sli_sp_pos = tf.expand_dims(sli_sp_pos, 2)
                sli_sp_pos = tf.tile(sli_sp_pos, [1,1,N_HIDDEN_STATES])



                #a_sp_p = tf.expand_dims(sp_p, 0)
                #a_sp_p = tf.expand_dims(a_sp_p, 0)
                #a_sp_p = tf.tile(a_sp_p, [len(t.struct[i]), N_HIDDEN_STATES, 1])
                #print("-------------------a_sp_p---------------------------------------------",a_sp_p)

                #to_sum = tf.multiply(a_sp_p, sli_var_a_up_ward)
                to_sum = tf.multiply(sli_sp_pos, sli_var_a_up_ward)

                #added = tf.reduce_sum(to_sum, [2])  # sommo nella dim 2
                added = tf.reduce_sum(to_sum, [1])  # sommo nella dim 2

                sli_in_prior = tf.transpose(sli_in_prior, perm=[1, 0])

                denominator = tf.multiply(sli_in_prior, added)
                denominator = tf.expand_dims(denominator, 1)
                denominator = tf.tile(denominator, [1, N_HIDDEN_STATES, 1])


                ris_24 = tf.divide(numerator, denominator)
                #ris_24 = tf.where(tf.is_inf(ris_24), tf.ones_like(ris_24), ris_24)
                #ris_24 = tf.where(tf.is_nan(ris_24), tf.zeros_like(ris_24), ris_24)

                test_numerator = numerator
                test_denominator = denominator
                #uniformarel la somma in moso che faccio uno su j+i (tutto)

                uniform = tf.reduce_sum(ris_24, [1,2])
                uniform = tf.expand_dims(uniform, 1)
                uniform = tf.expand_dims(uniform, 1)
                uniform = tf.tile(uniform, [1, N_HIDDEN_STATES,N_HIDDEN_STATES])
                ris_24 = tf.divide(ris_24, uniform)

                # univofrmare la somma su i e su j che faccia uno
                #uniform = tf.reduce_sum(ris_24, [1])
                #uniform = tf.expand_dims(uniform, 1)
                #uniform = tf.tile(uniform, [1, N_HIDDEN_STATES,1])
                #ris_24 = tf.divide(ris_24, uniform)
                #ris_24 = tf.where(tf.is_nan(ris_24), tf.zeros_like(ris_24), ris_24)



                #DDD

                # |||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||inglobe_ris_liv||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||

                head = tf.slice(var_EE, [0, 0, 0],
                                [t.struct[i][-1].name + 1 - int((ris_24.shape[0])), N_HIDDEN_STATES,
                                 N_HIDDEN_STATES])
                tail = tf.slice(var_EE, [t.struct[i][-1].name + 1, 0, 0],
                                [t.size - t.struct[i][-1].name - 1, N_HIDDEN_STATES, N_HIDDEN_STATES])

                ris_24 = tf.add(ris_24, tf.constant(1/10000000, dtype=tf.float64,shape=ris_24.shape))

                var_EE = tf.concat([head, ris_24, tail], 0)



                # |||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||computer25||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||

                ris_25 = tf.reduce_sum(ris_24, [2])


                #uniform = tf.reduce_sum(ris_25, [1])
                #uniform = tf.expand_dims(uniform, 1)
                #uniform = tf.tile(uniform, [1, N_HIDDEN_STATES])
                #ris_25 = tf.divide(ris_25, uniform)

                #ris_25 = tf.where(tf.is_nan(ris_25), tf.zeros_like(ris_25), ris_25)

                head = tf.slice(var_E, [0, 0], [t.struct[i][-1].name + 1 - int((ris_24.shape[0])),
                                                N_HIDDEN_STATES])  # _________________-da controllare la dim giusta in shape
                tail = tf.slice(var_E, [t.struct[i][-1].name + 1, 0],
                                [t.size - t.struct[i][-1].name - 1, N_HIDDEN_STATES])

                var_E = tf.concat([head, ris_25, tail], 0)

                # |||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||Reversed_Upward_Downward||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
            #uniform = tf.reduce_sum(var_EE, [1])
            #uniform = tf.expand_dims(uniform, 1)
            #uniform = tf.tile(uniform, [1, N_HIDDEN_STATES, 1])
            #var_EE = tf.divide(var_EE, uniform)
            #var_EE = tf.where(tf.is_nan(var_EE), tf.zeros_like(var_EE),var_EE)
           # var_EE = tf.add(var_EE, tf.constant(1/10000000, dtype=tf.float64,shape=var_EE.shape))
            #var_E = tf.add(var_E, tf.constant(1/10000000, dtype=tf.float64,shape=var_E.shape))


            #uniform = tf.reduce_sum(var_E, [1])
            #uniform = tf.expand_dims(uniform, 1)
            #uniform = tf.tile(uniform, [1, N_HIDDEN_STATES])
            #var_E = tf.divide(var_E, uniform)

            #print(" RUN ")

            var_EE_res,var_E_res ,ris_17_t= sess.run([var_EE,var_E,ris_17_t])
            
       # print("test_sliE",test_sliE)
       # print("test_sli_up_ward",test_sli_up_ward)
       # print("test_sli_sp_p",test_sli_sp_p)
       # print("test_sli_A",test_sli_A)

       # print("test_numerator",test_numerator)
       # print("test_denominator",test_denominator)
        var_EE_list.append(var_EE_res)
        var_E_list.append(var_E_res)        
        test1.append(var_a_up_ward)
        test2.append(var_up_ward)
        test3.append(ris_17_t)



        sess.close
        tf.reset_default_graph()
    #print("\nvar_EE_list\n",var_EE_list)
    #print("\var_E_list\n",var_E_list)   

    #print("\aux1\n",test_aux1)
    #print("\aux2\n",test_aux2)
    #print("\ris_17_t\n",test3)
        # |||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||M_step||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
        # |||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||M_step||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
        # |||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||M_step||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
        # |||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||M_step||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
        # |||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||M_step||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
        # |||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||M_step||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
        # |||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||M_step||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
        # |||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||M_step||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
        # |||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||M_step||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
        # |||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||M_step||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
        # |||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||M_step||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
        # |||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||M_step||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||

    #print(" M-step ")
    with tf.Session() as sess:


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
        # calcolo resized_tm_yu##
        #                                        DDDD   uesto può essere fatta in maniera migliore e più semplice andando a lavorare sugli indici
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

            test_resized_tm_yu =e_resized
            test_e_resized=e_resized

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
        #result_multinomial = tf.where(tf.is_inf(result_multinomial), tf.constant(1/N_HIDDEN_STATES, dtype=tf.float64,shape=result_multinomial.shape), result_multinomial)
        #result_multinomial = tf.where(tf.is_nan(result_multinomial),tf.constant(1/N_HIDDEN_STATES, dtype=tf.float64,shape=result_multinomial.shape), result_multinomial)
        #result_multinomial = tf.where(tf.less(result_multinomial,tf.constant(1/TO_ZERO, dtype=tf.float64,shape=result_multinomial.shape)), tf.zeros(dtype=tf.float64,shape=result_multinomial.shape), result_multinomial)

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
        #n_ii_list =N_HIDDEN_STATES * np.array(n_ii_list)
        #va bene niilist *3 ma non si fa cosi!
        result_sp = tf.divide(summed_sp, summed_sp_denominator)
        #result_sp = tf.divide(summed_sp, sum_N_I) #DDD se non e cosi elimina n_ii_list e tutto quello collegato precedentemente

        #result_sp = tf.where(tf.is_inf(result_sp), tf.constant(1/N_HIDDEN_STATES, dtype=tf.float64,shape=result_sp.shape), result_sp)
        #result_sp = tf.where(tf.is_nan(result_sp), tf.constant(1/N_HIDDEN_STATES, dtype=tf.float64,shape=result_sp.shape), result_sp)
        #result_sp = tf.where(tf.less(result_sp,tf.constant(1/TO_ZERO, dtype=tf.float64,shape=result_sp.shape)), tf.zeros(dtype=tf.float64,shape=result_sp.shape), result_sp)

        # STATE TRANSICTION (A)
        numerator_stat_tran = tf.reduce_sum(aux, [2,0])

        denominator_stat_tran = tf.reduce_sum(aux, [4, 2, 0])
        denominator_stat_tran = tf.expand_dims(denominator_stat_tran, 2)
        denominator_stat_tran = tf.tile(denominator_stat_tran, [1, 1, N_HIDDEN_STATES])

        result_state_trans = tf.divide(numerator_stat_tran, denominator_stat_tran)
        result_state_trans = tf.transpose(result_state_trans, [2, 1, 0])
        result_state_trans = tf.where(tf.is_inf(result_state_trans),tf.constant(1/N_HIDDEN_STATES, dtype=tf.float64,shape=result_state_trans.shape), result_state_trans)
        result_state_trans = tf.where(tf.is_nan(result_state_trans), tf.constant(1/N_HIDDEN_STATES, dtype=tf.float64,shape=result_state_trans.shape), result_state_trans)
        #result_state_trans = tf.where(tf.less(result_state_trans,tf.constant(1/TO_ZERO, dtype=tf.float64,shape=result_state_trans.shape)), tf.zeros(dtype=tf.float64,shape=result_state_trans.shape), result_state_trans)

        # PRIOR
        aux = tf.stack(aux_list_prior, 0)

        summed_prior = tf.reduce_sum(aux, [3, 2, 0])#DDD
        denominatore_summed_prior = tf.reduce_sum(summed_prior,[1])
        denominatore_summed_prior = tf.expand_dims(denominatore_summed_prior, 1)
        denominatore_summed_prior = tf.tile(denominatore_summed_prior, [1, N_HIDDEN_STATES])
        result_prior = tf.divide(summed_prior, denominatore_summed_prior)

        #result_prior = tf.where(tf.is_inf(result_prior), tf.zeros_like(result_prior), result_prior)
        #result_prior = tf.where(tf.is_nan(result_prior), tf.zeros_like(result_prior), result_prior)
        result_prior = tf.where(tf.is_inf(result_prior), tf.constant(1/N_HIDDEN_STATES, dtype=tf.float64,shape=result_prior.shape), result_prior)
        result_prior = tf.where(tf.is_nan(result_prior), tf.constant(1/N_HIDDEN_STATES, dtype=tf.float64,shape=result_prior.shape), result_prior)
        #result_prior = tf.where(tf.less(result_prior,tf.constant(1/TO_ZERO, dtype=tf.float64,shape=result_prior.shape)), tf.zeros(dtype=tf.float64,shape=result_prior.shape), result_prior)


        result_prior = tf.transpose(result_prior, [1, 0])

        #DDD devo far si che la somma suglii ia uno?

        pi = result_prior
        sp_p = result_sp
        A = result_state_trans
        bi = result_multinomial

        # |||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||       fine        M_step      ||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
        # |||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||       fine        M_step      ||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
        # |||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||       fine        M_step      ||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
        # |||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||       fine        M_step      ||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
        # |||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||       fine        M_step      ||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||

        # |||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||M_stlog_likelihoodep||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
        #print(" LIKELIHOOD")

        tot = 0
        s1=0
        s2=0
        s3=0
        s4=0
        for i in range(0, len(data_set)):

            # prelevo i nodi interni e foglia
            leaf_node = tf.slice(var_E_list[i], [data_set[i].struct[-1][0].name, 0],
                                 [data_set[i].size - data_set[i].struct[-1][0].name, N_HIDDEN_STATES])

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
            #log_bi = tf.where(tf.less(log_bi,tf.constant(1/TO_ZERO, dtype=tf.float64,shape=log_bi.shape)), tf.zeros(dtype=tf.float64,shape=log_bi.shape), log_bi)

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
            #log_sp_p = tf.where(tf.less(log_sp_p,tf.constant(1/TO_ZERO, dtype=tf.float64,shape=log_sp_p.shape)), tf.zeros(dtype=tf.float64,shape=log_sp_p.shape), log_sp_p)

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
            #quarta_somm = tf.where(tf.less(quarta_somm,tf.constant(1/TO_ZERO, dtype=tf.float64,shape=quarta_somm.shape)), tf.zeros(dtype=tf.float64,shape=quarta_somm.shape), quarta_somm)

            tot = tot + prima_somm + seconda_somm + terza_somm + quarta_somm
            s1= s1+prima_somm
            s2=s2+seconda_somm
            s3=s3+terza_somm
            s4=s4+quarta_somm
        #print(" RUN m step + log_likelihood")
        pi,sp_p,bi,A,tot ,s1,s2,s3 , s4 = sess.run([result_prior,result_sp,result_multinomial,result_state_trans,tot, s1 , s2 , s3 , s4])
        
        s_1.append(s1)
        s_2.append(s2)
        s_3.append(s3)
        s_4.append(s4)
        like_list.append(tot)


        #if (np.isnan(tot)):
         #   break;
          #  break;
           # print(" IL LIKELIHOOD  è Nan")
        
       

        #print("prima seconda ternza 4",seconda_somm, log_bi, var_EE_list)
        #print("\n\n\n\n\n\n")
        t_pi=np.sum(pi,0)
        t_sp=np.sum(sp_p,0)
        t_a=np.sum(A,0)
        t_bi=np.sum(bi,1)

        #print("var_EE_list",var_EE_list[0].shape)
        #print("var_EE_list",var_EE_list)
        #print("test_auxx",test_auxx.shape)
       # print("test_auxx",test_auxx)
      #  print(" summed_prior  \n",summed_prior)
     #   print(" n_l_list  \n",n_l_list)
        #print("  var_EE_list  ",var_EE_list)

        #print("PARAMETRI\n")
        #print("pi",pi)
        #print("A",A)
        #print("bi",bi)
        #print("sp_p",sp_p)
        #print("VINCOLI DA RISPETTARE\n")
        #print(" t_pi  ",t_pi)
        #print(" t_sp  ",t_sp)
        #print(" t_a  ",t_a)
        #print(" t_bi  ",t_bi)

        sess.close
    tf.reset_default_graph()



        # |||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||           tlog_likelihood         fine            ||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
    # |||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||   fine            M_stlog_likelihoodep||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||





#with tf.Session() as sess:
 #   init = tf.global_variables_initializer()
  #  sess.run(init)
    #print(like_list)
    #print(sess.run([aux,  pi,t_pi,  sp_p,t_sp_p,  A,t_A,  bi,t_bi]))
   # a = sess.run([like_list])

    #print(a)
##||||||||||||||||||||||||||||||||||||||||||||||LOGLIKEHOLD||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
#print(like_list)

#pl.plot(like_list)
#pl.plot(s_4,color='red')
#pl.plot(s_3,color='blue')
#pl.plot(s_2,color='orange')
#pl.plot(s_1,color='green')
#pl.plot(like_list)


np.savetxt('ilike_list.out', like_list) 
np.savetxt('is1.out', s_1) 
np.savetxt('is2.out', s_2) 
np.savetxt('is3.out', s_3) 
np.savetxt('is4.out', s_4) 
#pl.show()
#pl.savefig('10.png')

#||||||||||||||||||||||||||||||||||||||||||||||||||||||||E-STEP||||||||||||||||||||||||||||||||||||||||||||||||||||||||


