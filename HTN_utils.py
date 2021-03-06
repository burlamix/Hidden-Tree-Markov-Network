import numpy as np
import tensorflow as tf
import math
from BHTMM import *

MAX_CHILD = 66
N_SYMBOLS = 66

def init_theta(hidden_state,empty=False):

	th =[[],[],[],[]]
	if(empty == False):

		th[0] =   np.random.random((hidden_state,hidden_state, MAX_CHILD)) 		#a
		th[1] =   np.random.random((MAX_CHILD))									# sp_p
		th[2] =   np.random.random((hidden_state, MAX_CHILD))					# pi
		th[3] =   np.random.random((hidden_state, N_SYMBOLS))					# bi
	else:
		th[0] =   np.zeros((hidden_state,hidden_state, MAX_CHILD)) 		
		th[1] =   np.zeros((MAX_CHILD))									
		th[2] =   np.zeros((hidden_state, MAX_CHILD))					
		th[3] =   np.zeros((hidden_state, N_SYMBOLS))					

	return th


def init_contrastive_matrix(shape, dtype=None):
	m_init = np.zeros(shape, dtype=dtype)
	p=0
	s=1
	for i in range(0,shape[1]):
		m_init[p,i]=1
		m_init[s,i]=-1
		if(s==shape[0]-1):
			p=p+1
			s=p
		s=s+1

	return m_init

def softmax_for_all(ph_sp_p, ph_a, ph_bi, ph_pi,hidden_state):

	sf_a = tf.exp(ph_a) / tf.add(\
							tf.reduce_sum(\
								tf.exp(ph_a),\
										0), \
								tf.constant(1/10000000, dtype=tf.float64,shape=ph_a.shape))

	sf_sp_p = tf.exp(ph_sp_p) / tf.add(\
									tf.reduce_sum(\
										tf.exp(ph_sp_p),\
												 0), \
									tf.constant(1/10000000, dtype=tf.float64,shape=ph_sp_p.shape))	

	sf_bi = tf.exp(ph_bi) / tf.add(\
								tf.tile(\
									tf.expand_dims(\
										tf.reduce_sum(tf.exp(ph_bi), 1),\
													1) \
										,[1,N_SYMBOLS]), \
								tf.constant(1/10000000, dtype=tf.float64,shape=ph_bi.shape))

	sf_pi = tf.exp(ph_pi) / tf.add(\
								tf.tile(\
									tf.expand_dims(\
										tf.reduce_sum(tf.exp(ph_pi), 0),\
													0)\
										,[hidden_state,1]),\
								tf.constant(1/10000000, dtype=tf.float64,shape=ph_pi.shape))


	#sf_bi = tf.add(sf_bi, tf.constant(1/10000000, dtype=tf.float64,shape=sf_bi.shape))
	#sf_a = tf.add(sf_a, tf.constant(1/10000000, dtype=tf.float64,shape=sf_a.shape))
	#sf_sp_p = tf.add(sf_sp_p, tf.constant(1/10000000, dtype=tf.float64,shape=sf_sp_p.shape))
	#sf_pi = tf.add(sf_pi, tf.constant(1/10000000, dtype=tf.float64,shape=sf_pi.shape))
	return sf_sp_p, sf_a, sf_bi, sf_pi

def param_update(tot_delta_sp_p, tot_delta_a, tot_delta_bi, tot_delta_pi,sf_sp_p, sf_a, sf_bi, sf_pi,lerning_rate,var_EE_list,var_E_list,hidden_state,t,batch_size,j,last):

	lista_n_in_ee = [[] for i in range (MAX_CHILD)]
	lista_n_in_e = [[] for i in range (MAX_CHILD)]
	lista_n_in_e_sp = [[] for i in range (MAX_CHILD)]
	lista_n_foglia = [[] for i in range (MAX_CHILD)]
	lista_symbol = [[] for i in range (N_SYMBOLS)]
	lista_tv = [[] for i in range (N_SYMBOLS)]
	max_l = -1
	max_s = -1
	max_p = -1
	for_pad = tf.zeros([1,hidden_state],dtype=tf.float64)

	#aggiungo una riga di zeri in fondo per prelevare con la gather un termine inerte
	var_E_prov= tf.concat([var_E_list,for_pad],0)
	#in modo da rendere piu veloce l  esecuzione mi salvo in una lista ordinatamente i nodi e le lero posizioni


	#prelevo il nomo dei nodi che sono l esimi figli
	for level in t.struct[:-1]:
		for nodo in level:
			for child_n in range(0, len(nodo.children)):
				lista_n_in_ee[child_n].append(nodo.children[child_n].name)
				lista_n_in_e[child_n].append(nodo.children[child_n].name)

	for internal_list in lista_n_in_ee:
		if max_l < len(internal_list):
			max_l=len(internal_list)

	# uniformo la lunghezza cosi da non rendere il tensore sparso per la futura gather
	for k  in range(0, len(lista_n_in_ee)):
		start = len(lista_n_in_ee[k])
		for kk in range(start, int(max_l)):
			lista_n_in_ee[k].append(0)
			lista_n_in_e[k].append(t.size)
	
	#-----------------------A------------------

	#prelevo i valori di interesse dalle variabili
	slice_ee = tf.gather(var_EE_list, lista_n_in_ee)
	slice_e = tf.gather(var_E_prov, lista_n_in_e)

	#uniformo la dimensione di slice_e
	slice_e = tf.expand_dims(slice_e, 2)
	slice_e = tf.tile(slice_e, [1, 1, hidden_state, 1])
	#duplico a per il numero di nodi lesimi massimo
	a_aux = tf.expand_dims(sf_a, 3)
	a_aux = tf.tile(a_aux, [1, 1, 1, int(max_l)])			
	slice_ee = tf.transpose(slice_ee, [2,3,0,1])
	slice_e = tf.transpose(slice_e, [2,3,0,1])



	to_sub = tf.multiply(slice_e, a_aux)
	to_sum = tf.subtract(slice_ee, to_sub)
	delta_a = tf.reduce_sum(to_sum,[3])

	#-----------------------pi------------------
	for node in t.struct[-1]:
		lista_n_foglia[nodo.pos-1].append(nodo.name)

	for internal_list_sp in lista_n_foglia:
		if max_p < len(internal_list_sp):
			max_p=len(internal_list_sp)

	# uniformo la lunghezza cosi da non rendere il tensore sparso per la futura gather
	for k  in range(0, len(lista_n_foglia)):
		start = len(lista_n_foglia[k])
		for kk in range(start, int(max_p)):
			lista_n_foglia[k].append(t.size)

	slice_e = tf.gather(var_E_prov, lista_n_foglia)
	slice_e = tf.reduce_sum(slice_e,1)
	slice_e = tf.transpose(slice_e,[1,0])
	to_sub = tf.multiply(sf_pi, len(t.struct[-1]))

	delta_pi = tf.subtract(slice_e,to_sub)

	#-----------------------bi------------------

	np_tm_yu = np.zeros([int(t.size), N_SYMBOLS])

	for level in t.struct:
		for node in level:
			np_tm_yu[node.name, node.label] = 1





	#var_E_list lo espando per 367
	e_aux = tf.expand_dims(var_E_list, 2)
	e_aux = tf.tile(e_aux, [1, 1, N_SYMBOLS])
	e_aux = tf.transpose(e_aux, [1,0,2])

	bi_aux = tf.expand_dims(sf_bi, 1)
	bi_aux = tf.tile(bi_aux, [1, t.size, 1])

	tm_yu = tf.expand_dims(np_tm_yu, 0)
	tm_yu = tf.tile(tm_yu, [hidden_state,1, 1])

	to_mul = tf.subtract(tm_yu,bi_aux)
	to_sum = tf.multiply(e_aux, to_mul)

	delta_bi = tf.reduce_sum(to_sum,[1])

	#-----------------------sp_p------------------

	slice_ee = tf.gather(var_EE_list, lista_n_in_ee)
	slice_e = tf.reduce_sum(slice_ee,[2,3])	


	sp_p_aux = tf.expand_dims(sf_sp_p, 1)
	sp_p_aux = tf.tile(sp_p_aux, [1, max_l ])				

	da_sottrarre = tf.multiply(tf.cast(t.N_I,tf.float64),sp_p_aux)


	to_sum = tf.subtract(slice_e,da_sottrarre)

	delta_sp_p = tf.reduce_sum(to_sum,[1])


	#aggiorno il delta del gradiente
	tot_delta_bi   = tot_delta_bi   +  delta_bi 	
	tot_delta_pi   = tot_delta_pi   +  delta_pi 	
	tot_delta_a    = tot_delta_a    +  delta_a 	
	tot_delta_sp_p = tot_delta_sp_p +  delta_sp_p 


	return tot_delta_sp_p, tot_delta_a, tot_delta_bi, tot_delta_pi


#funzione che calcola le combinazioni di n elementi di classe r
def n_comb_r(n,r):
    f = math.factorial
    return int(f(n) / f(r) / f(n-r))
