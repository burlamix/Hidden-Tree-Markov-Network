import numpy as np
#import scipy.misc as sc
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Input, Dense
from keras.models import Model
from keras import backend as bk
#from GPU_E_M_utils import *
from E_M_utils import *
from utils_keras_g import *

from operator import add
import math

import time
from datetime import timedelta

np.random.seed(42)


M=30
K=11
N_SYMBOLS = 367
MAX_CHILD = 32


lerning_rate=0.01
epoche=25
hidden_state = 10
batch_size = 50

#cl_size = sc.comb(M, 2).astype(np.int64)
cl_size = nCr(M,2)


#FILE2 = "inex05.train.elastic.tree"
#FILE2 = "test_1000.tree"
#FILE2 = "test_500.tree"
FILE2 = "test_100.tree"

data_set = dataset_parser(FILE2)



model = Sequential()
model.add(Dense(cl_size, activation='tanh',trainable=False,kernel_initializer=my_init2, input_dim=M))
model.add(Dense(K, activation='softmax'))
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])



#inizializzo random i parametri del modello
free_th_l = [init_theta_old(hidden_state) for i in range(M)] 

#contiene i valori della batch per l'aggioramento del gradiente
delta_th = [init_theta_zero(hidden_state) for i in range(M)] 



for i in range (0,epoche):
	print("EPOCA: ",i)

	# per ogni epoca analizzo tutto il dataset

	like_list_aux = np.zeros((batch_size,M), dtype=np.float64)
	one_hot_lab = np.zeros((batch_size,K), dtype=np.float64)

	for j in range(0,len(data_set)):
		#print("albero: ",j)

		like_list=[]
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
			#th_l = softmax_for_all_old(free_th_l[x_m],hidden_state)
			sf_sp_p, sf_a, sf_bi, sf_pi = softmax_for_all_old(ph_sp_p, ph_a, ph_bi, ph_pi,hidden_state)

			#E-STEP
			var_EE, var_E = Reversed_Upward_Downward(sf_sp_p, sf_a, sf_bi, sf_pi, data_set[j], hidden_state)

			#LIKELIHOOD
			like = log_likelihood_test(sf_pi,sf_sp_p,sf_a,sf_bi,var_EE,var_E,data_set[j],hidden_state)

			#AGGIORNO I PARAMETRI 
			new_sp_p, new_a, new_bi, new_pi  = param_update(delta_sp_p, delta_a, delta_bi, delta_pi, ph_sp_p, ph_a, ph_bi, ph_pi, sf_sp_p, sf_a, sf_bi, sf_pi, lerning_rate,var_EE,var_E,hidden_state,data_set[j],batch_size,j)

			
			for k in range(M):
				delta_th[k][1], delta_th[k][3], delta_th[k][2], delta_th[k][0], xlike = sess.run([new_sp_p, new_bi, new_pi, new_a,like],{ ph_a: free_th_l[k][0] , ph_sp_p: free_th_l[k][1] ,ph_bi: free_th_l[k][3], ph_pi: free_th_l[k][2], delta_a: delta_th[k][0] , delta_sp_p: delta_th[k][1] , delta_bi: delta_th[k][3], delta_pi: delta_th[k][2]}) 
				like_list.append(xlike)
			sess.close()


		#metto la lista dei vaori di likelihood nella lista che verrà appasata come batch
		like_list_aux[j%batch_size]=like_list
		#crea la lista come vuole keras per l'obbiettivo
		one_hot_lab[j%batch_size][int(data_set[j].classe)-1]=1

		if( j%batch_size == batch_size-1):
			#print("				aggiorno gradiente")
			#aggiorno i lambe free già calcolati
			free_th_l = delta_th

			#model.train_on_batch(like_list_aux,one_hot_lab)
			model.fit(like_list_aux, one_hot_lab, epochs=1)
			#res = model.predict(like_list_aux)
			like_list_aux = np.zeros((batch_size,M), dtype=np.float64)
			one_hot_lab = np.zeros((batch_size,K), dtype=np.float64)
			delta_th = [init_theta_zero(hidden_state) for i in range(M)] 

			tf.reset_default_graph()

		

			#print(res)



#model.fit(data, one_hot_labels, epochs=1000, batch_size=32)




# Train the model, iterating on the data in batches of 32 samples


