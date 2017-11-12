from keras_t_g import *
import random

#	PARAMETRI DEL MODELLO

m=40

hidden_state = 10

lerning_rate=0.01

decay=1e-6

epoche = 20

batch_size = 32

stop_n = 5


#TRANIG SET
traning_set = "data/inex05.train.elastic.tree"

#traning_set = "data/3_train.tree"
#traning_set = "data/128_train.tree"
#traning_set = "data/train_666.tree"
#traning_set = "data/train_100.tree"


#TEST SET
test_set = "data/inex05.test.elastic.tree"

#test_set = "data/3_test.tree"
#test_set = "data/128_train.tree"
#test_set = "data/test_666.tree"
#test_set = "data/test_100.tree"



#VALIDATION
#vali_set = "data/test_250.tree"

#Train senza validation
#traning_set = dataset_parser(traning_set)
#random.shuffle(traning_set)

test_set = dataset_parser(test_set)
random.shuffle(test_set)


#train con 3-validation
traning_set = dataset_parser_class(traning_set)
traning_set = divide_tre_validation_htm(traning_set)


modello = HTM(m,lerning_rate,decay)


#result = train_and_test(modello,hidden_state,m,lerning_rate,epoche,batch_size,traning_set[0])

#htm , lamda = training (modello,hidden_state,m,lerning_rate,epoche,batch_size,traning_set,decay,stop_n,batch_size)
htm , lamda = training_val (modello,hidden_state,m,lerning_rate,epoche,batch_size,traning_set[0][0],decay,stop_n,traning_set[0][1],batch_size)

print("test...")

result 		= test(htm,lamda,test_set,m,hidden_state)





print("\n\n          FINAL  LOSS AND ACCURACY ON TEST           ",result)
print("\n\n")


