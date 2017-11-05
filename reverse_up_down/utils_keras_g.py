import numpy as np
import scipy.misc as sc
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Input, Dense
from keras.models import Model
from keras import backend as bk
from GPU_E_M_utils import *
#from E_M_utils import *

from keras.activations import softmax

import multiprocessing
from functools import partial
import math

MAX_CHILD=32
N_SYMBOLS = 367


class theta:
	def __init__(self,hidden_state,empty=False):
		if(empty == False):
			self[0] =    np.random.random((hidden_state,hidden_state, MAX_CHILD))
			self[1] = np.random.random((MAX_CHILD))
			self[2] =   np.random.random((hidden_state, MAX_CHILD))
			self[3] =   np.random.random((hidden_state, N_SYMBOLS))
		else:
			self[0] =    None
			self[1] = None
			self[2] =    None
			self[3] =   None

def init_theta_old(hidden_state,empty=False):

	th =[[],[],[],[]]
	if(empty == False):

		th[0] =   np.random.random((hidden_state,hidden_state, MAX_CHILD)) 		#a
		th[1] =   np.random.random((MAX_CHILD))									# sp_p
		th[2] =   np.random.random((hidden_state, MAX_CHILD))					# pi
		th[3] =   np.random.random((hidden_state, N_SYMBOLS))					# bi

	return th


def init_theta(hidden_state,empty=False):

	th =[[],[],[],[]]
	if(empty == False):

		th[0] =   random_sum_one3(0, hidden_state, hidden_state, MAX_CHILD) 	#a
		th[1] =   random_sum_one1(MAX_CHILD)									# sp_p
		th[2] =   random_sum_one2(0, hidden_state, MAX_CHILD)					# pi
		th[3] =   random_sum_one2(1, hidden_state, N_SYMBOLS)					# bi

	return th

def my_init2(shape, dtype=None):
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


def softmax_for_all_old(ph_sp_p, ph_a, ph_bi, ph_pi,hidden_state):

	new_th_l = init_theta(hidden_state,empty=True) 

	sf_a= tf.nn.softmax(ph_a,dim=0)
	sf_sp_p = tf.nn.softmax(ph_sp_p)
	sf_pi = tf.nn.softmax(ph_pi,dim=1)
	sf_bi = tf.nn.softmax(ph_bi,dim=0)

	return sf_sp_p, sf_a, sf_bi, sf_pi

def softmax_for_all(th_l,hidden_state):

	new_th_l = [init_theta(hidden_state,empty=True) for i in range(len(th_l))] 

	for i in range(len(th_l)):


		#new_th_l[i][0] = tf.divide( tf.exp(new_th_l[i][0]) , tf.reduce_sum(tf.exp(new_th_l[i][0]), 0))

		new_th_l[i][0] = tf.nn.softmax(th_l[i][0],dim=0)
		new_th_l[i][1] = tf.nn.softmax(th_l[i][1])
		new_th_l[i][3] = tf.nn.softmax(th_l[i][3],dim=1)
		new_th_l[i][2] = tf.nn.softmax(th_l[i][2],dim=0)

	return new_th_l



def param_update(ph_sp_p, ph_a, ph_bi, ph_pi,sf_sp_p, sf_a, sf_bi, sf_pi,lerning_rate,var_EE_list,var_E_list,hidden_state,t):



	#print("	  up_m: ",j)

	lista_n_in_ee = [[] for i in range (MAX_CHILD)]
	lista_n_in_e = [[] for i in range (MAX_CHILD)]
	lista_n_foglia = [[] for i in range (MAX_CHILD)]
	lista_symbol = [[] for i in range (N_SYMBOLS)]
	lista_tv = [[] for i in range (N_SYMBOLS)]
	max_l = -1
	for_pad = np.zeros([1,hidden_state])

	#aggiungo una riga di zeri in fondo per prelevare con la gather un termine inerte
	#var_E_prov= np.concatenate((var_E_list,for_pad))
	var_E_prov= tf.concat([var_E_list,for_pad],0)

	#in modo da rendere piu veloce l  esecuzione mi salvo in una lista ordinatamente i nodi e le lero posizioni
	for l_number in t.N_L:
		if max_l < l_number:
			max_l = l_number
	max_l=max_l+1

	#prelevo il nomo dei nodi che sono l esimi figli
	for level in t.struct[:-1]:
		for nodo in level:
			for child_n in range(0, len(nodo.children)):
				lista_n_in_ee[child_n].append(nodo.children[child_n].name)
				lista_n_in_e[child_n].append(nodo.children[child_n].name)

	# uniformo la lunghezza cosi da non rendere il tensore sparso per la futura gather
	for k  in range(0, len(lista_n_in_ee)):
		start = len(lista_n_in_ee[k])
		for kk in range(start, int(max_l)):
			lista_n_in_ee[k].append(0)
			lista_n_in_e[k].append(t.size)

	#prelevo i valori di interesse dalle variabili
	slice_ee = tf.gather(var_EE_list, lista_n_in_ee)
	slice_e = tf.gather(var_E_prov, lista_n_in_e)

	#uniformo la dimensione di slice_e
	slice_e = tf.expand_dims(slice_e, 2)
	slice_e = tf.tile(slice_e, [1, 1, hidden_state, 1])

	#duplico a per il numero di nodi lesimi massimo
	a_aux = tf.expand_dims(sf_a, 3)
	a_aux = tf.tile(a_aux, [1, 1, 1, int(max_l)])			
		
	slice_ee = tf.transpose(slice_ee, [2,3,0,1])#--------------------------------------------DDDD???? ij
	slice_e = tf.transpose(slice_e, [2,3,0,1])

	to_sub = tf.multiply(slice_e, a_aux)
	to_sum = tf.multiply(slice_ee, -(to_sub))
	delta_a = tf.reduce_sum(to_sum,[3])

	#-----------------------pi------------------

	#prelevo il nomo dei nodi che sono l esimi figli
	for nodo in t.struct[-1]:
		lista_n_foglia[nodo.pos-1].append(nodo.name)

	# uniformo la lunghezza cosi da non rendere il tensore sparso per la futura gather
	for k  in range(0, len(lista_n_foglia)):
		start = len(lista_n_foglia[k])
		for kk in range(start, int(max_l)):
			lista_n_foglia[k].append(t.size)

	slice_e = tf.gather(var_E_prov, lista_n_foglia)
	slice_e = tf.transpose(slice_e, [2,1,0])

	pi_aux = tf.expand_dims(sf_pi, 1)
	pi_aux = tf.tile(pi_aux, [1, int( max_l), 1])

	to_sub = tf.multiply(pi_aux, len(t.struct[-1]))
	to_sum = tf.multiply(slice_e, -(to_sub))
	delta_pi = tf.reduce_sum(to_sum,[1])


	#-----------------------bi------------------

	tm_yu = np.zeros([int(t.size), N_SYMBOLS])

	for level in t.struct:
		for node in level:
			tm_yu[node.name, node.label] = 1

	#var_E_list lo espando per 367
	e_aux = tf.expand_dims(var_E_list, 2)
	e_aux = tf.tile(e_aux, [1, 1, N_SYMBOLS])
	e_aux = tf.transpose(e_aux, [1,0,2])

	bi_aux = tf.expand_dims(sf_bi, 1)
	bi_aux = tf.tile(bi_aux, [1, t.size, 1])

	tm_yu = tf.expand_dims(tm_yu, 0)
	tm_yu = tf.tile(tm_yu, [hidden_state,1, 1])

	to_mul = tf.subtract(tm_yu,bi_aux)
	to_sum = tf.multiply(e_aux, to_mul)

	delta_bi = tf.reduce_sum(to_sum,[1])

	#-----------------------sp_p------------------
	internal_e = tf.slice(var_E_list, [0, 0], [t.struct[-1][0].name, hidden_state])

	e_aux = tf.expand_dims(internal_e, 2)
	e_aux = tf.tile(e_aux, [1, 1, MAX_CHILD])
	#e_aux = tf.transpose(e_aux, [1,0,2])

	sp_p_aux = tf.expand_dims(sf_sp_p, 0)
	sp_p_aux = tf.expand_dims(sp_p_aux, 0)
	sp_p_aux = tf.tile(sp_p_aux, [t.N_I, hidden_state,1])				


	sp_p_aux = tf.multiply(tf.cast(t.N_I,tf.float64),sp_p_aux)
	delta_sp_p = tf.subtract(e_aux,sp_p_aux)
	delta_sp_p = tf.reduce_sum(delta_sp_p,[0,1])



	ph_bi   = ph_bi + lerning_rate*delta_bi
	ph_pi   = ph_pi + lerning_rate*delta_pi
	ph_a    = ph_a + lerning_rate*delta_a
	ph_sp_p = ph_sp_p+ lerning_rate*delta_sp_p


	return ph_sp_p, ph_a, ph_bi, ph_pi



def E_step_like(th_l,t,hidden_state):

	# e qui che puo essere fatto multithreading!!!!!!
		#print("	  e_m: ",j)

	var_EE, var_E = Reversed_Upward_Downward(th_l[1], th_l[0], th_l[3], th_l[2], t, hidden_state)

	like = log_likelihood_test(th_l[2],th_l[1],th_l[0],th_l[3],var_EE,var_E,t,hidden_state)

	return [var_EE,var_E,like]










def param_update_m(free_th_l,th_l,lerning_rate,var_EE_list,var_E_list,hidden_state,t,m):


	for j in range(0,m):

		#print("	  up_m: ",j)

		lista_n_in_ee = [[] for i in range (MAX_CHILD)]
		lista_n_in_e = [[] for i in range (MAX_CHILD)]
		lista_n_foglia = [[] for i in range (MAX_CHILD)]
		lista_symbol = [[] for i in range (N_SYMBOLS)]
		lista_tv = [[] for i in range (N_SYMBOLS)]
		max_l = -1
		for_pad = np.zeros([1,hidden_state])

		#aggiungo una riga di zeri in fondo per prelevare con la gather un termine inerte
		#var_E_prov= np.concatenate((var_E_list[j],for_pad))
		var_E_prov= tf.concat([var_E_list[j],for_pad],0)

		#in modo da rendere piu veloce l  esecuzione mi salvo in una lista ordinatamente i nodi e le lero posizioni
		for l_number in t.N_L:
			if max_l < l_number:
				max_l = l_number
		max_l=max_l+1

		#prelevo il nomo dei nodi che sono l esimi figli
		for level in t.struct[:-1]:
			for nodo in level:
				for child_n in range(0, len(nodo.children)):
					lista_n_in_ee[child_n].append(nodo.children[child_n].name)
					lista_n_in_e[child_n].append(nodo.children[child_n].name)

    	# uniformo la lunghezza cosi da non rendere il tensore sparso per la futura gather
		for k  in range(0, len(lista_n_in_ee)):
			start = len(lista_n_in_ee[k])
			for kk in range(start, int(max_l)):
				lista_n_in_ee[k].append(0)
				lista_n_in_e[k].append(t.size)

		#prelevo i valori di interesse dalle variabili
		slice_ee = tf.gather(var_EE_list[j], lista_n_in_ee)
		slice_e = tf.gather(var_E_prov, lista_n_in_e)

		#uniformo la dimensione di slice_e
		slice_e = tf.expand_dims(slice_e, 2)
		slice_e = tf.tile(slice_e, [1, 1, hidden_state, 1])

		#duplico th_l[j][0] per il numero di nodi lesimi massimo
		a_aux = tf.expand_dims(th_l[j][0], 3)
		a_aux = tf.tile(a_aux, [1, 1, 1, int(max_l)])			
			
		slice_ee = tf.transpose(slice_ee, [2,3,0,1])#--------------------------------------------DDDD???? ij
		slice_e = tf.transpose(slice_e, [2,3,0,1])

		to_sub = tf.multiply(slice_e, a_aux)
		to_sum = tf.multiply(slice_ee, -(to_sub))
		delta_a = tf.reduce_sum(to_sum,[3])

		#-----------------------pi------------------

		#prelevo il nomo dei nodi che sono l esimi figli
		for nodo in t.struct[-1]:
			lista_n_foglia[nodo.pos-1].append(nodo.name)

    	# uniformo la lunghezza cosi da non rendere il tensore sparso per la futura gather
		for k  in range(0, len(lista_n_foglia)):
			start = len(lista_n_foglia[k])
			for kk in range(start, int(max_l)):
				lista_n_foglia[k].append(t.size)

		slice_e = tf.gather(var_E_prov, lista_n_foglia)
		slice_e = tf.transpose(slice_e, [2,1,0])

		pi_aux = tf.expand_dims(th_l[j][2], 1)
		pi_aux = tf.tile(pi_aux, [1, int( max_l), 1])

		to_sub = tf.multiply(pi_aux, len(t.struct[-1]))
		to_sum = tf.multiply(slice_e, -(to_sub))
		delta_pi = tf.reduce_sum(to_sum,[1])


		#-----------------------bi------------------

		tm_yu = np.zeros([int(t.size), N_SYMBOLS])

		for level in t.struct:
			for node in level:
				tm_yu[node.name, node.label] = 1

		#var_E_list[j] lo espando per 367
		e_aux = tf.expand_dims(var_E_list[j], 2)
		e_aux = tf.tile(e_aux, [1, 1, N_SYMBOLS])
		e_aux = tf.transpose(e_aux, [1,0,2])

		bi_aux = tf.expand_dims(th_l[j][3], 1)
		bi_aux = tf.tile(bi_aux, [1, t.size, 1])

		tm_yu = tf.expand_dims(tm_yu, 0)
		tm_yu = tf.tile(tm_yu, [hidden_state,1, 1])

		to_mul = tf.subtract(tm_yu,bi_aux)
		to_sum = tf.multiply(e_aux, to_mul)

		delta_bi = tf.reduce_sum(to_sum,[1])

		#-----------------------sp_p------------------
		internal_e = tf.slice(var_E_list[j], [0, 0], [t.struct[-1][0].name, hidden_state])

		e_aux = tf.expand_dims(internal_e, 2)
		e_aux = tf.tile(e_aux, [1, 1, MAX_CHILD])
		#e_aux = tf.transpose(e_aux, [1,0,2])

		sp_p_aux = tf.expand_dims(th_l[j][1], 0)
		sp_p_aux = tf.expand_dims(sp_p_aux, 0)
		sp_p_aux = tf.tile(sp_p_aux, [t.N_I, hidden_state,1])				


		sp_p_aux = tf.multiply(tf.cast(t.N_I,tf.float64),sp_p_aux)
		delta_sp_p = tf.subtract(e_aux,sp_p_aux)
		delta_sp_p = tf.reduce_sum(delta_sp_p,[0,1])



		free_th_l[j][3] = free_th_l[j][3] +lerning_rate*delta_bi
		free_th_l[j][2] = free_th_l[j][2] +lerning_rate*delta_pi
		free_th_l[j][0] = free_th_l[j][0] +lerning_rate*delta_a
		free_th_l[j][1] = free_th_l[j][1] +lerning_rate*delta_sp_p


	return free_th_l




def E_step_like_m(th_l,t,m,hidden_state):

	var_EE_list =[]
	var_E_list =[]
	like_list =[]


	# e qui che puo essere fatto multithreading!!!!!!
	for j in range(0,m):

		#print("	  e_m: ",j)


		var_EE, var_E = Reversed_Upward_Downward(th_l[j][1], th_l[j][0], th_l[j][3], th_l[j][2], t, hidden_state)

		var_EE_list.append(var_EE)

		var_E_list.append(var_E)


		like = log_likelihood_test(th_l[j][2],th_l[j][1],th_l[j][0],th_l[j][3],var_EE,var_E,t,hidden_state)

		like_list.append(like)




	return [var_EE_list,var_E_list,like_list]


def nCr(n,r):
    f = math.factorial
    return int(f(n) / f(r) / f(n-r))