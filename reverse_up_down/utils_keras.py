import numpy as np
import scipy.misc as sc
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Input, Dense
from keras.models import Model
from keras import backend as bk
from E_M_utils import *
from keras.activations import softmax

import multiprocessing
from functools import partial

MAX_CHILD=32
N_SYMBOLS = 367


class theta:
	def __init__(self,hidden_state,empty=False):
		if(empty == False):
			self.a =    np.random.random((hidden_state,hidden_state, MAX_CHILD))
			self.sp_p = np.random.random((MAX_CHILD))
			self.pi =   np.random.random((hidden_state, MAX_CHILD))
			self.bi =   np.random.random((hidden_state, N_SYMBOLS))
		else:
			self.a =    None
			self.sp_p = None
			self.pi =    None
			self.bi =   None

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


def softmax_for_all(th_l,hidden_state):

	g_0 = tf.Graph()
	with g_0.as_default():

		new_th_l = [theta(hidden_state,empty=True) for i in range(len(th_l))] 

		for i in range(len(th_l)):
			with tf.Session() as sess:	#

				sp_p = tf.nn.softmax(th_l[i].sp_p)
				a = tf.nn.softmax(th_l[i].a,dim=0)
				bi = tf.nn.softmax(th_l[i].bi,dim=1)
				pi = tf.nn.softmax(th_l[i].pi,dim=0)

				new_th_l[i].sp_p,new_th_l[i].a,new_th_l[i].bi,new_th_l[i].pi = sess.run([sp_p,a,bi,pi])

			sess.close

	return new_th_l



def param_update(free_th_l,th_l,lerning_rate,var_EE_list,var_E_list,hidden_state,t,m):

	g_2 = tf.Graph()

	with g_2.as_default():

		for j in range(0,m):
			print("	  up_m: ",j)

			with tf.Session() as sess:


				lista_n_in_ee = [[] for i in range (MAX_CHILD)]
				lista_n_in_e = [[] for i in range (MAX_CHILD)]
				lista_n_foglia = [[] for i in range (MAX_CHILD)]
				lista_symbol = [[] for i in range (N_SYMBOLS)]
				lista_tv = [[] for i in range (N_SYMBOLS)]
				max_l = -1
				for_pad = np.zeros([1,hidden_state])


				#aggiungo una riga di zeri in fondo per prelevare con la gather un termine inerte
				var_E_prov= np.concatenate((var_E_list[j],for_pad))

				#in modo da rendere più veloce l'esecuzione mi salvo in una lista ordinatamente i nodi e le lero posizioni
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

				#duplico th_l[j].a per il numero di nodi lesimi massimo
				a_aux = tf.expand_dims(th_l[j].a, 3)
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

				pi_aux = tf.expand_dims(th_l[j].pi, 1)
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

				bi_aux = tf.expand_dims(th_l[j].bi, 1)
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

				sp_p_aux = tf.expand_dims(th_l[j].sp_p, 0)
				sp_p_aux = tf.expand_dims(sp_p_aux, 0)
				sp_p_aux = tf.tile(sp_p_aux, [t.N_I, hidden_state,1])				


				sp_p_aux = tf.multiply(tf.cast(t.N_I,tf.float64),sp_p_aux)
				delta_sp_p = tf.subtract(e_aux,sp_p_aux)
				delta_sp_p = tf.reduce_sum(delta_sp_p,[0,1])



				new_bi = free_th_l[j].bi +lerning_rate*delta_bi
				new_pi = free_th_l[j].pi +lerning_rate*delta_pi
				new_a = free_th_l[j].a +lerning_rate*delta_a
				new_sp_p = free_th_l[j].sp_p +lerning_rate*delta_sp_p


				free_th_l[j].bi,free_th_l[j].pi,free_th_l[j].a,free_th_l[j].sp_p = sess.run([new_bi,new_pi,new_a,new_sp_p])

				#print("internal_node_ee",leaf_node_ee)
				#print("slice_e",slice_e)
				sess.close

	return free_th_l


def E_step_like(th_l,t,m,hidden_state):

	var_EE_list =[]
	var_E_list =[]
	like_list = np.zeros((1,m), dtype=np.float64)

	g_1 = tf.Graph()

	with g_1.as_default():

		# è qui che può essere fatto multithreading!!!!!!
		for j in range(0,m):
			print("	  e_m: ",j)

			with tf.Session() as sess:

				var_EE, var_E = Reversed_Upward_Downward(th_l[j].sp_p, th_l[j].a, th_l[j].bi, th_l[j].pi, t, hidden_state)

				var_EE,var_E = sess.run([var_EE,var_E])
				var_EE_list.append(var_EE)
				var_E_list.append(var_E)

				sess.close
			

			with tf.Session() as sess:

				like = log_likelihood_test(th_l[j].pi,th_l[j].sp_p,th_l[j].a,th_l[j].bi,var_EE,var_E,t,hidden_state)
				like = sess.run(like)
				like_list[0][j]=like

				sess.close

	return [var_EE_list,var_E_list,like_list]




def E_step_like_multi(th_l,t,m,hidden_state):

	var_EE_list =[]
	var_E_list =[]
	like_list = np.zeros((1,m), dtype=np.float64)




	iterable = range(m)

	pool = multiprocessing.Pool(m+12)
	func = partial(E_step_like_esec, th_l, t, hidden_state)

	print("dopo partial")

	result = pool.map(func, iterable)

	print("dopo pool")

	pool.join()

	pool.close()

	print("risultati-",result)

	return  results

def E_step_like_esec(th_l,t,hidden_state,j):

	print("buongiorno",j)

	#g_1 = tf.Graph()

	with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:


		var_EE, var_E = Reversed_Upward_Downward(th_l[j].sp_p, th_l[j].a, th_l[j].bi, th_l[j].pi, t, hidden_state)

		print("prima run ",j)

		print(sess.run([var_EE,var_E]))

		print("dopo run",j)

		sess.close
	

	with tf.Session() as sess:

		like = log_likelihood_test(th_l[j].pi,th_l[j].sp_p,th_l[j].a,th_l[j].bi,var_EE,var_E,t,hidden_state)
		like = sess.run(like)

		sess.close
	
	print("ciao!",j)

	return [var_EE,var_E,like]





'''



tm_yu = np.zeros([int(t.size), N_SYMBOLS])

for level in t.struct:
	for node in level:
		tm_yu[node.name, node.label] = 1

for level in t.struct:
	for nodo in level:						
		lista_symbol[nodo.label].append(nodo.name)

		lista_symbol[nodo.label].append(nodo.name)
max=-1
for sub_list in lista_symbol:
	if(max< len(sub_list)):
		max = len(sub_list)

for k  in range(0, len(lista_symbol)):
	start = len(lista_symbol[k])
	for kk in range(start, max):
		lista_symbol[k].append(t.size)

tv_xu = np.zeros([max, N_SYMBOLS])
for z1 in range(max):
	for z2 in range(N_SYMBOLS):
		if


slice_e = tf.gather(var_E_list[j], lista_symbol)
slice_e = tf.transpose(slice_e, [2,1,0])

bi_aux = tf.expand_dims(th_l[j].bi, 1)
bi_aux = tf.tile(bi_aux, [1, int(max), 1])
'''