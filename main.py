from keras_t_g import *
import random

#	PARAMETRI DEL MODELLO

m=30
hidden_state = 10

lerning_rate=0.01
decay=1e-6

epoche = 10

batch_size = 1

stop_n = 5

#TRANIG SET
traning_set = "data/inex05.train.elastic.tree"
#traning_set = "data/test_1.tree"
#traning_set = "data/test_100.tree"
#traning_set = "data/test_2.tree"

#TEST SET
test_set = "data/inex05.test.elastic.tree"
#test_set = "data/test_1000.tree"
#test_set = "data/test_666.tree"
#test_set = "data/test_1.tree"

#VALIDATION
#vali_set = "data/validation_666.tree"

#Train senza validation
traning_set = dataset_parser(traning_set)
random.shuffle(traning_set)

test_set = dataset_parser(test_set)
random.shuffle(test_set)


#vali_set = dataset_parser(vali_set)
#random.shuffle(vali_set)

#train con 3-validation
#data_train = dataset_parser_class(traning_set)
#data_train = divide_tre_validation_htm(data_train)


modello = HTM(m,lerning_rate,decay)




#result = train_and_test(modello,hidden_state,m,lerning_rate,epoche,batch_size,data_train[0])

htm , lamda = training (modello,hidden_state,m,lerning_rate,epoche,batch_size,traning_set,decay,stop_n)
#htm , lamda = training_val (modello,hidden_state,m,lerning_rate,epoche,batch_size,data_train,decay,stop_n,vali_set)

print("test...")

result 		= test(htm,lamda,test_set,m,hidden_state)





print("\n\n          FINAL  LOSS AND ACCURACY ON TEST           ",result)
print("\n\n")


