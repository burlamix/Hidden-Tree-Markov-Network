import numpy as np
import scipy.misc as sc
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Input, Dense
from keras.models import Model
from keras import backend as bk
from E_M_utils import *
from utils_keras import *
np.random.seed(42)


M=5
K=11
N_SYMBOLS = 367
MAX_CHILD = 32


lerning_rate=0.5
epoche=10
hidden_state = 10

cl_size = sc.comb(M, 2).astype(np.int64)


FILE2 = "test_4.tree"

data_set = dataset_parser(FILE2)


print("nome processo",__name__ )



model = Sequential()
model.add(Dense(cl_size, activation='tanh',trainable=False,kernel_initializer=my_init2, input_dim=M))
model.add(Dense(K, activation='softmax'))
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])



#inizializzo random i parametri del modello
free_th_l = [theta(hidden_state) for i in range(M)] 



for i in range (0,epoche):
	print("EPOCA: ",i)

	# per ogni epoca analizzo tutto il dataset
	for j in range(0,len(data_set)):
		print("  albero: ",j)

		#normalizzo i parametri del modello
		th_l = softmax_for_all(free_th_l,hidden_state)


		#calcolo E_step e il loglikelihood 
		var_EE_list,var_E_list,like_list = E_step_like_multi(th_l,data_set[j],M,hidden_state)

		#codifico la classe risultato
		one_hot_lab = np.zeros((1,K), dtype=np.float64)
		one_hot_lab[0][int(data_set[j].classe)-1]=1

		#model.train_on_batch(like_list,one_hot_labels)

		model.fit(like_list, one_hot_lab, epochs=1)

		free_th_l = param_update(free_th_l,th_l,lerning_rate,var_EE_list,var_E_list,hidden_state,data_set[j],M)


		#res = model.predict(like_list)



#model.fit(data, one_hot_labels, epochs=1000, batch_size=32)




# Train the model, iterating on the data in batches of 32 samples


