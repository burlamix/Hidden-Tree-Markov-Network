from keras_t_g import *


#	PARAMETRI DEL MODELLO

m=30

lerning_rate=0.01

epoche = 2

hidden_state = 10

batch_size = 9

#TRANIG SET
#traning_set = "inex05.train.elastic.tree"
traning_set = "data/train_100.tree"
#traning_set = "data/train_100.tree"

#TEST SET
#test_set = "inex05.train.elastic.tree"
test_set = "data/test_100.tree"
#test_set = "data/test_100.tree"

data_train = dataset_parser(traning_set)

data_test = dataset_parser(test_set)


modello = HTM(m)



htm , lamda = training(modello,hidden_state,m,lerning_rate,epoche,batch_size,data_train)




result 		= test(htm,lamda,data_test,m,hidden_state)


print("\n\n                         ",result)