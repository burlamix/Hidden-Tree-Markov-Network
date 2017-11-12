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


from keras.callbacks import LearningRateScheduler

#import pylab as pl


np.set_printoptions(threshold=np.nan)

nome_file = "hope_tv_b32_ep20_lr_01"

#classi

K=11
MAX_CHILD = 32
N_SYMBOLS = 367

lr_global=0.01

def step_decay(epoch):

	return float(lr_global)

def HTM (m,lerning_rate,dec):

	cl_size = nCr(m,2)

	model = Sequential()
	model.add(Dense(cl_size, activation= 'tanh' ,trainable=False,kernel_initializer=init_contrastive_matrix, input_dim=m))
	model.add(Dense(K, activation= 'softmax' ))

	
	sgd = optimizers.SGD(lr=lerning_rate, decay=dec, momentum=0.5)
	#sgd = keras.optimizers.RMSprop(lr=lerning_rate, rho=0.9, epsilon=1e-08, decay=0.0)
	#sgd = keras.optimizers.Adadelta(lr=lerning_rate, rho=0.95, epsilon=1e-08, decay=0.0)
	#sgd = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

	model.compile(optimizer=sgd ,
	              loss= 'categorical_crossentropy' ,
	              metrics=[ 'accuracy' ])
	return model






def training_val(htm,hidden_state,m,lerning_rate,epoche,batch_size,data_set,decay,stop_n,vali_set,bs):


	lrate = LearningRateScheduler(step_decay)


	plot_list_loss=[]
	plot_list_acc=[]

	stop_var=-1
	count_stop=0
	#calcolo la dimensione del primo livello di nodi interno

	#inizializzo random i parametri del modello
	free_th_l = [init_theta(hidden_state) for i in range(m)] 

	#contiene i valori della batch per l aggioramento del gradiente
	delta_th = [init_theta_zero(hidden_state) for i in range(m)] 

	global lr_global
	lr_global=lerning_rate

	for i in range (0,epoche):
	
		print("EPOCA: ",i)

		#ordino in modo casuale il dataset
		random.shuffle(data_set)

		like_list_aux = []
		one_hot_lab = []


		#traning 
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
					new_sp_p, new_a, new_bi, new_pi = param_update(delta_sp_p, delta_a, delta_bi, delta_pi, sf_sp_p, sf_a, sf_bi, sf_pi, lerning_rate,var_EE,var_E,hidden_state,data_set[j],batch_size,j,len(data_set)-1)
					

					#CALCOLO IL TUTTO
					for k in range(m):
						delta_th[k][1], delta_th[k][3], delta_th[k][2], delta_th[k][0], xlike = sess.run([new_sp_p, new_bi, new_pi, new_a,like],{ ph_a: free_th_l[k][0] , ph_sp_p: free_th_l[k][1] ,ph_bi: free_th_l[k][3], ph_pi: free_th_l[k][2], delta_a: delta_th[k][0] , delta_sp_p: delta_th[k][1] , delta_bi: delta_th[k][3], delta_pi: delta_th[k][2]}) 
						like_list.append(xlike)

					sess.close()

			#metto la lista dei vaori di likelihood nella lista che verra appasata come batch
			like_list_aux.append(like_list)
			zeri_k=np.zeros(K, dtype=np.float64)
			zeri_k[int(data_set[j].classe)-1]=1
			one_hot_lab.append(zeri_k)

			if( j%batch_size == batch_size-1 or j ==len(data_set)-1):

				dime = len(like_list_aux)
				like_list_aux =np.array(like_list_aux )
				one_hot_lab =np.array(one_hot_lab )

				for z in range(m):
					free_th_l[z][0]   = free_th_l[z][0] +  ((lerning_rate)*(delta_th[z][0]/dime))
					free_th_l[z][1]   = free_th_l[z][1] +  ((lerning_rate)*(delta_th[z][1]/dime))
					free_th_l[z][2]   = free_th_l[z][2] +  ((lerning_rate)*(delta_th[z][2]/dime))
					free_th_l[z][3]   = free_th_l[z][3] +  ((lerning_rate)*(delta_th[z][3]/dime))

				#aggiorno il gradente dei parametri dei HTMM

				#p = htm.train_on_batch(like_list_aux,one_hot_lab)
				p =	htm.fit( like_list_aux, one_hot_lab, batch_size=bs, epochs=1, verbose=0, callbacks=[lrate])
				#print(p)
				like_list_aux = []
				one_hot_lab = []
				delta_th = [init_theta_zero(hidden_state) for zz in range(m)] 
	

		lerning_rate = lerning_rate * (1. / (1. + (decay * (i+1))))
		lr_global =lerning_rate

		like_list_epoca= np.zeros((len(vali_set),m), dtype=np.float64)
		one_hot_lab_epoca = np.zeros((len(vali_set),K), dtype=np.float64)
		
		#print("validation")
		#CALCOLO LOSS SUL VALIDATION
		for j in range(0,len(vali_set)):
			
			#print("     tree: ",j)

			like_list_v=[]

			with tf.Graph().as_default():

				with tf.Session() as sess:

					ph_a = tf.placeholder(shape=[hidden_state, hidden_state, MAX_CHILD], dtype=tf.float64)
					ph_sp_p = tf.placeholder(shape=[MAX_CHILD], dtype=tf.float64)
					ph_bi = tf.placeholder(shape=[hidden_state, N_SYMBOLS], dtype=tf.float64)
					ph_pi = tf.placeholder(shape=[hidden_state, MAX_CHILD], dtype=tf.float64)

					#NORMALIZZO I PARAMETRI
					sf_sp_p, sf_a, sf_bi, sf_pi = softmax_for_all(ph_sp_p, ph_a, ph_bi, ph_pi,hidden_state)

					#E-STEP
					var_EE, var_E = Reversed_Upward_Downward(sf_sp_p, sf_a, sf_bi, sf_pi, vali_set[j], hidden_state)

					#LIKELIHOOD
					like = log_likelihood_test(sf_pi,sf_sp_p,sf_a,sf_bi,var_EE,var_E,vali_set[j],hidden_state)
					

					#CALCOLO IL TUTTO
					for k in range(m):
						xlike = sess.run(like,{ ph_a: free_th_l[k][0] , ph_sp_p: free_th_l[k][1] ,ph_bi: free_th_l[k][3], ph_pi: free_th_l[k][2]}) 
						like_list_v.append(xlike)

					sess.close()


			#valori per il test sull'epoca
			like_list_epoca[j]=like_list_v
			one_hot_lab_epoca[j][int(vali_set[j].classe)-1]=1

		loss_function,accuracy = htm.test_on_batch(like_list_epoca,one_hot_lab_epoca)

		print("                     loss = ",loss_function,"   ac =",accuracy)

		with open(nome_file, "a") as myfile:
		    myfile.write(str(loss_function)+";"+str(accuracy)+"\n")

		plot_list_loss.append(loss_function)
		plot_list_acc.append(accuracy)

		#EARLY STOPPING
		if(accuracy > stop_var):
			stop_var=accuracy
		#if(loss_function < stop_var):
		#	stop_var=loss_function
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







def training(htm,hidden_state,m,lerning_rate,epoche,batch_size,data_set,decay,stop_n,bs):

	lrate = LearningRateScheduler(step_decay)

	global lr_global
	lr_global=lerning_rate

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

		like_list_aux = []
		one_hot_lab = []

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
					new_sp_p, new_a, new_bi, new_pi = param_update(delta_sp_p, delta_a, delta_bi, delta_pi, sf_sp_p, sf_a, sf_bi, sf_pi, lerning_rate,var_EE,var_E,hidden_state,data_set[j],batch_size,j,len(data_set)-1)
					

					#CALCOLO IL TUTTO
					for k in range(m):
						delta_th[k][1], delta_th[k][3], delta_th[k][2], delta_th[k][0], xlike = sess.run([new_sp_p, new_bi, new_pi, new_a,like],{ ph_a: free_th_l[k][0] , ph_sp_p: free_th_l[k][1] ,ph_bi: free_th_l[k][3], ph_pi: free_th_l[k][2], delta_a: delta_th[k][0] , delta_sp_p: delta_th[k][1] , delta_bi: delta_th[k][3], delta_pi: delta_th[k][2]}) 
						like_list.append(xlike)

					sess.close()


			#metto la lista dei vaori di likelihood nella lista che verra appasata come batch
			like_list_aux.append(like_list)
			zeri_k=np.zeros(K, dtype=np.float64)
			zeri_k[int(data_set[j].classe)-1]=1
			one_hot_lab.append(zeri_k)


			#valori per il test sull'epoca
			like_list_epoca[j]=like_list
			one_hot_lab_epoca[j][int(data_set[j].classe)-1]=1

			if( j%batch_size == batch_size-1 or j ==len(data_set)-1):

				dime = len(like_list_aux)
				like_list_aux =np.array(like_list_aux )
				one_hot_lab =np.array(one_hot_lab )

				#aggiorno il gradente dei parametri dei HTMM
				for z in range(m):
					free_th_l[z][0]   = free_th_l[z][0] +  ((lerning_rate)*(delta_th[z][0]/dime))
					free_th_l[z][1]   = free_th_l[z][1] +  ((lerning_rate)*(delta_th[z][1]/dime))
					free_th_l[z][2]   = free_th_l[z][2] +  ((lerning_rate)*(delta_th[z][2]/dime))
					free_th_l[z][3]   = free_th_l[z][3] +  ((lerning_rate)*(delta_th[z][3]/dime))

				
				#p = htm.train_on_batch(like_list_aux,one_hot_lab)
				p =	htm.fit( like_list_aux, one_hot_lab, batch_size=bs, epochs=1, verbose=0, callbacks=[lrate])

				like_list_aux=[]
				one_hot_lab=[]

				delta_th = [init_theta_zero(hidden_state) for i in range(m)] 
		
		lerning_rate = lerning_rate * (1. / (1. + (decay * (i+1))))
		lr_global =lerning_rate
		
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