import numpy as np
#import scipy.misc as sc
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Input, Dense
from keras.models import Model
from keras import backend as bk
from GPU_E_M_utils import *
#from E_M_utils import *
from utils_keras_g import *


import math

import time
from datetime import timedelta

np.random.seed(42)


M=50
K=11
N_SYMBOLS = 367
MAX_CHILD = 32


lerning_rate=0.2
epoche=1
hidden_state = 2

#cl_size = sc.comb(M, 2).astype(np.int64)
cl_size = nCr(M,2)


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


print(free_th_l[0][0].shape)

for i in range (0,epoche):
	print("EPOCA: ",i)

	# per ogni epoca analizzo tutto il dataset
	for j in range(0,len(data_set)):

		like_list=[]
		with tf.Session() as sess:

			ph_a = tf.placeholder(shape=[hidden_state, hidden_state, MAX_CHILD], dtype=tf.float64)
			ph_sp_p = tf.placeholder(shape=[MAX_CHILD], dtype=tf.float64)
			ph_bi = tf.placeholder(shape=[hidden_state, N_SYMBOLS], dtype=tf.float64)
			ph_pi = tf.placeholder(shape=[hidden_state, MAX_CHILD], dtype=tf.float64)

			#NORMALIZZO I PARAMETRI
			#th_l = softmax_for_all_old(free_th_l[x_m],hidden_state)
			sf_sp_p, sf_a, sf_bi, sf_pi = softmax_for_all_old(ph_sp_p, ph_a, ph_bi, ph_pi,hidden_state)

			#E-STEP
			var_EE, var_E = Reversed_Upward_Downward(sf_sp_p, sf_a, sf_bi, sf_pi, data_set[j], hidden_state)

			#LIKELIHOOD
			like = log_likelihood_test(sf_pi,sf_sp_p,sf_a,sf_bi,var_EE,var_E,data_set[j],hidden_state)

			#AGGIORNO I PARAMETRI 
			new_sp_p, new_a, new_bi, new_pi  = param_update(ph_sp_p, ph_a, ph_bi, ph_pi, sf_sp_p, sf_a, sf_bi, sf_pi, lerning_rate,var_EE,var_E,hidden_state,data_set[j])

			
			for i in range(M):
				free_th_l[i][1], free_th_l[i][3], free_th_l[i][2], free_th_l[i][0], xlike = sess.run([new_sp_p, new_bi, new_pi, new_a,like],{ ph_a: free_th_l[i][0] , ph_sp_p: free_th_l[i][1] ,ph_bi: free_th_l[i][3], ph_pi: free_th_l[i][2] })
				#print(xlike)
				like_list.append(xlike)
			sess.close()




		one_hot_lab = np.zeros((1,K), dtype=np.float64)
		one_hot_lab[0][int(data_set[j].classe)-1]=1
		#tf.reset_default_graph()

		like_list_aux = np.zeros((1,M), dtype=np.float64)
		like_list_aux[0]=like_list

		#model.train_on_batch(like_list_aux,one_hot_lab)
		model.fit(like_list_aux, one_hot_lab, epochs=1)
		tf.reset_default_graph()

		
		#res = model.predict(like_list)



#model.fit(data, one_hot_labels, epochs=1000, batch_size=32)




# Train the model, iterating on the data in batches of 32 samples


