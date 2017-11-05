import numpy as np
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.models import Model
from utils_keras_g import *
#from GPU_E_M_utils import *
from E_M_utils import *


K=11
MAX_CHILD = 32
N_SYMBOLS = 367



def HTM (m):

	cl_size = nCr(m,2)

	model = Sequential()
	model.add(Dense(cl_size, activation= 'tanh' ,trainable=False,kernel_initializer=init_contrastive_matrix, input_dim=m))
	model.add(Dense(K, activation= softmax ))
	model.compile(optimizer= rmsprop ,
	              loss= categorical_crossentropy ,
	              metrics=[ accuracy ])
	return model

def training(htm,hidden_state,m,lerning_rate,epoche,batch_size,data_set):

	#calcolo la dimensione del primo livello di nodi interno

	#inizializzo random i parametri del modello
	free_th_l = [init_theta(hidden_state) for i in range(m)] 

	#contiene i valori della batch per l aggioramento del gradiente
	delta_th = [init_theta_zero(hidden_state) for i in range(m)] 


	for i in range (0,epoche):

		#print("EPOCA: ",i)

		like_list_aux = np.zeros((batch_size,m), dtype=np.float64)
		one_hot_lab = np.zeros((batch_size,K), dtype=np.float64)

		for j in range(0,len(data_set)):
			
			#print("     tree: ",j)

			like_list=[]

			with tf.Graph().as_default():

				with tf.Session() as sess:

					ph_a = tf.placeholder(shape=[hidden_state, hidden_state, MAX_CHILD], dtype=tf.float64)
					ph_sp_p = tf.placeholder(shape=[MAX_CHILD], dtype=tf.float64)
					ph_bi = tf.placeholder(shape=[hidden_state, N_SYMBOLS], dtype=tf.float64)
					ph_pi = tf.placeholder(shape=[hidden_state, MAX_CHILD], dtype=tf.float64)

					delta_a = tf.placeholder(shape=[hidden_state, hidden_state, MAX_CHILD], dtype=tf.float64)
					delta_sp_p = tf.placeholder(shape=[MAX_CHILD], dtype=tf.float64)
					delta_bi = tf.placeholder(shape=[hidden_state, N_SYMBOLS], dtype=tf.float64)
					delta_pi = tf.placeholder(shape=[hidden_state, MAX_CHILD], dtype=tf.float64)

					#NORMALIZZO I PARAMETRI
					sf_sp_p, sf_a, sf_bi, sf_pi = softmax_for_all(ph_sp_p, ph_a, ph_bi, ph_pi,hidden_state)

					#E-STEP
					var_EE, var_E = Reversed_Upward_Downward(sf_sp_p, sf_a, sf_bi, sf_pi, data_set[j], hidden_state)

					#LIKELIHOOD
					like = log_likelihood_test(sf_pi,sf_sp_p,sf_a,sf_bi,var_EE,var_E,data_set[j],hidden_state)

					#AGGIORNO I PARAMETRI 
					new_sp_p, new_a, new_bi, new_pi  = param_update(delta_sp_p, delta_a, delta_bi, delta_pi, ph_sp_p, ph_a, ph_bi, ph_pi, sf_sp_p, sf_a, sf_bi, sf_pi, lerning_rate,var_EE,var_E,hidden_state,data_set[j],batch_size,j)
					

					#CALCOLO IL TUTTO
					for k in range(m):
						delta_th[k][1], delta_th[k][3], delta_th[k][2], delta_th[k][0], xlike = sess.run([new_sp_p, new_bi, new_pi, new_a,like],{ ph_a: free_th_l[k][0] , ph_sp_p: free_th_l[k][1] ,ph_bi: free_th_l[k][3], ph_pi: free_th_l[k][2], delta_a: delta_th[k][0] , delta_sp_p: delta_th[k][1] , delta_bi: delta_th[k][3], delta_pi: delta_th[k][2]}) 
						like_list.append(xlike)

					sess.close()


			#metto la lista dei vaori di likelihood nella lista che verra appasata come batch
			like_list_aux[j%batch_size]=like_list
			#crea la lista come vuole keras per l obbiettivo
			one_hot_lab[j%batch_size][int(data_set[j].classe)-1]=1

			if( j%batch_size == batch_size-1):

				#aggiorno il gradente dei parametri dei HTMM
				free_th_l = delta_th

				p = htm.train_on_batch(like_list_aux,one_hot_lab)
				#DDDD con questo puoi farci il grafico

				#htm.fit(like_list_aux,one_hot_lab,epochs=1)

				like_list_aux = np.zeros((batch_size,m), dtype=np.float64)
				one_hot_lab = np.zeros((batch_size,K), dtype=np.float64)

				delta_th = [init_theta_zero(hidden_state) for i in range(m)] 







	return htm , free_th_l


def test(htm,free_th_l,data_set,m,hidden_state):


	like_list_aux = np.zeros((len(data_set),m), dtype=np.float64)
	one_hot_lab = np.zeros((len(data_set),K), dtype=np.float64)


	for j in range(0,len(data_set)):
		
		print("albero: ",j)

		like_list=[]

		with tf.Graph().as_default():

			with tf.Session() as sess:

				ph_a = tf.placeholder(shape=[hidden_state, hidden_state, MAX_CHILD], dtype=tf.float64)
				ph_sp_p = tf.placeholder(shape=[MAX_CHILD], dtype=tf.float64)
				ph_bi = tf.placeholder(shape=[hidden_state, N_SYMBOLS], dtype=tf.float64)
				ph_pi = tf.placeholder(shape=[hidden_state, MAX_CHILD], dtype=tf.float64)

				#NORMALIZZO I PARAMETRI
				sf_sp_p, sf_a, sf_bi, sf_pi = softmax_for_all(ph_sp_p, ph_a, ph_bi, ph_pi,hidden_state)

				#E-STEP
				var_EE, var_E = Reversed_Upward_Downward(sf_sp_p, sf_a, sf_bi, sf_pi, data_set[j], hidden_state)

				#LIKELIHOOD
				like = log_likelihood_test(sf_pi,sf_sp_p,sf_a,sf_bi,var_EE,var_E,data_set[j],hidden_state)


				#CALCOLO IL TUTTO
				for k in range(m):
					xlike = sess.run(like,{ ph_a: free_th_l[k][0] , ph_sp_p: free_th_l[k][1] ,ph_bi: free_th_l[k][3], ph_pi: free_th_l[k][2]}) 
					like_list.append(xlike)

				sess.close()

		#metto la lista dei vaori di likelihood nella lista che verra appasata come batch
		like_list_aux[j]=like_list
		#crea la lista come vuole keras per l obbiettivo
		one_hot_lab[j][int(data_set[j].classe)-1]=1

	result = htm.test_on_batch(like_list_aux,one_hot_lab)

	return result
