import numpy as np
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.models import Model
from utils_keras_g import *
#from GPU_E_M_utils import *
from E_M_utils import *
from keras import optimizers
from keras.models import load_model
import h5py
from keras import initializers





#import pylab as pl



np.set_printoptions(threshold=np.nan)

nome_file = "last_12_sdg_b1_01"

#classi
K=11
MAX_CHILD = 32
N_SYMBOLS = 367



def HTM (m,lerning_rate,dec):

	cl_size = nCr(m,2)

	model = Sequential()
	model.add(Dense(cl_size, activation= 'tanh' ,trainable=False,kernel_initializer=init_contrastive_matrix, input_dim=m))
	model.add(Dense(K, activation= 'softmax' ))

	
	sgd = optimizers.SGD(lr=lerning_rate, decay=dec, momentum=0.5)
	#sgd = keras.optimizers.RMSprop(lr=lerning_rate, rho=0.9, epsilon=1e-08, decay=0.0)
	#sgd = keras.optimizers.Adadelta(lr=lerning_rate, rho=0.95, epsilon=1e-08, decay=0.0)
	model.compile(optimizer=sgd ,
	              loss= 'categorical_crossentropy' ,
	              metrics=[ 'accuracy' ])
	
	
	return model






def training(htm,hidden_state,m,lerning_rate,epoche,batch_size,data_set,decay,stop_n):

	plot_list_loss=[]
	plot_list_acc=[]

	stop_var=99999999999999
	count_stop=0
	#calcolo la dimensione del primo livello di nodi interno

	#inizializzo random i parametri del modello
	free_th_l = [init_theta(hidden_state) for i in range(m)] 

	#contiene i valori della batch per l aggioramento del gradiente
	delta_th = [init_theta_zero(hidden_state) for i in range(m)] 


	for i in range (0,epoche):

		#print("EPOCA: ",i)

		#ordino in modo casuale il dataset
		random.shuffle(data_set)

		like_list_aux = np.zeros((batch_size,m), dtype=np.float64)
		one_hot_lab = np.zeros((batch_size,K), dtype=np.float64)

		like_list_epoca= np.zeros((len(data_set),m), dtype=np.float64)
		one_hot_lab_epoca = np.zeros((len(data_set),K), dtype=np.float64)
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
					new_sp_p, new_a, new_bi, new_pi = param_update(delta_sp_p, delta_a, delta_bi, delta_pi, ph_sp_p, ph_a, ph_bi, ph_pi, sf_sp_p, sf_a, sf_bi, sf_pi, lerning_rate,var_EE,var_E,hidden_state,data_set[j],batch_size,j,len(data_set)-1)
					

					#CALCOLO IL TUTTO
					for k in range(m):
						delta_th[k][1], delta_th[k][3], delta_th[k][2], delta_th[k][0], xlike = sess.run([new_sp_p, new_bi, new_pi, new_a,like],{ ph_a: free_th_l[k][0] , ph_sp_p: free_th_l[k][1] ,ph_bi: free_th_l[k][3], ph_pi: free_th_l[k][2], delta_a: delta_th[k][0] , delta_sp_p: delta_th[k][1] , delta_bi: delta_th[k][3], delta_pi: delta_th[k][2]}) 
						like_list.append(xlike)

					sess.close()


			#metto la lista dei vaori di likelihood nella lista che verra appasata come batch
			like_list_aux[j%batch_size]=like_list
			#crea la lista come vuole keras per l obbiettivo
			one_hot_lab[j%batch_size][int(data_set[j].classe)-1]=1

			#valori per il test sull'epoca
			like_list_epoca[j]=like_list
			one_hot_lab_epoca[j][int(data_set[j].classe)-1]=1

			if( j%batch_size == batch_size-1 or j ==len(data_set)-1):

				#aggiorno il gradente dei parametri dei HTMM
				free_th_l = delta_th

				lerning_rate = lerning_rate * (1. / (1. + (decay * i)))
				
				p = htm.train_on_batch(like_list_aux,one_hot_lab)

				#print("		batch		",p)

				#htm.fit(like_list_aux,one_hot_lab,epochs=1)

				like_list_aux = np.zeros((batch_size,m), dtype=np.float64)
				one_hot_lab = np.zeros((batch_size,K), dtype=np.float64)

				delta_th = [init_theta_zero(hidden_state) for i in range(m)] 
	
		loss_function,accuracy = htm.test_on_batch(like_list_epoca,one_hot_lab_epoca)

		print("        loss = ",loss_function,"   ac =",accuracy)

		with open(nome_file, "a") as myfile:
		    myfile.write(str(loss_function)+";"+str(accuracy)+"\n")

		plot_list_loss.append(loss_function)
		plot_list_acc.append(accuracy)

		#EARLY STOPPING
		if(loss_function < stop_var):
			stop_var=loss_function
			count_stop=0
			htm.save_weights("weights_"+nome_file)

		else:
			count_stop = count_stop +1

		if(count_stop==stop_n):
			print("STOP")
			break




	np.savetxt("plot_loss_"+nome_file, plot_list_loss) 
	np.savetxt("plot_acc_"+nome_file, plot_list_acc) 

	#pl.plot(plot_list_loss)
	#pl.plot(plot_list_acc)
	#pl.show()

	htm.load_weights("weights_"+nome_file)

	return htm , free_th_l

def test(htm,free_th_l,data_set,m,hidden_state):


	like_list_aux = np.zeros((len(data_set),m), dtype=np.float64)
	one_hot_lab = np.zeros((len(data_set),K), dtype=np.float64)


	for j in range(0,len(data_set)):
		
		#print("albero: ",j)

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


def train_and_test(modello,hidden_state,m,lerning_rate,epoche,batch_size,data_train):

	htm , lamda = training(modello,hidden_state,m,lerning_rate,epoche,batch_size,data_train[0])

	result 		= test(htm,lamda,data_train[1],m,hidden_state)

	return result