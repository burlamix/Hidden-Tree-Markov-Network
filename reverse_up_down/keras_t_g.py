import numpy as np
import scipy.misc as sc
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Input, Dense
from keras.models import Model
from keras import backend as bk
from E_M_utils import *
from utils_keras_g import *



import time
from datetime import timedelta

np.random.seed(42)


M=20
K=11
N_SYMBOLS = 367
MAX_CHILD = 32


lerning_rate=0.5
epoche=25
hidden_state = 10

cl_size = sc.comb(M, 2).astype(np.int64)


FILE2 = "test_100.tree"

data_set = dataset_parser(FILE2)


print("nome processo",__name__ )



model = Sequential()
model.add(Dense(cl_size, activation='tanh',trainable=False,kernel_initializer=my_init2, input_dim=M))
model.add(Dense(K, activation='softmax'))
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])



#inizializzo random i parametri del modello
free_th_l = [init_theta(hidden_state) for i in range(M)] 



for i in range (0,epoche):
	print("EPOCA: ",i)

	# per ogni epoca analizzo tutto il dataset
	for j in range(0,len(data_set)):

		g_1 = tf.Graph()
		with g_1.as_default():

			with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
		
				print("  albero: ",j)

				#normalizzo i parametri del modello
				th_l = softmax_for_all(free_th_l,hidden_state)


				#calcolo E_step e il loglikelihood 
				var_EE_list,var_E_list,like_list = E_step_like(th_l,data_set[j],M,hidden_state)


				#calcolo i nuovi parametri
				free_th_l = param_update(free_th_l,th_l,lerning_rate,var_EE_list,var_E_list,hidden_state,data_set[j],M)


				#print("run")				
				#like_list , free_th_l = sess.run([like_list,free_th_l])
				start_time = time.monotonic()

				like_list ,free_th_l = sess.run([like_list,free_th_l])

				end_time = time.monotonic()
				#print(timedelta(seconds=end_time - start_time))

				sess.close()



		one_hot_lab = np.zeros((1,K), dtype=np.float64)
		one_hot_lab[0][int(data_set[j].classe)-1]=1
		#tf.reset_default_graph()

		like_list_aux = np.zeros((1,M), dtype=np.float64)
		like_list_aux[0]=like_list

		model.train_on_batch(like_list_aux,one_hot_lab)
		#model.fit(like_list_aux, one_hot_lab, epochs=1)

		
		#res = model.predict(like_list)



#model.fit(data, one_hot_labels, epochs=1000, batch_size=32)




# Train the model, iterating on the data in batches of 32 samples


