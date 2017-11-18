from keras_t_g import *
import random

#	PARAMETRI DEL MODELLO

m=40
hidden_state = 6
lerning_rate = 0.01
decay=1e-6
epoche = 50
batch_size = 32
stop_n = 6

nome_file = "final_v1_m40_c6"

modello = HTM(m,lerning_rate,decay)

#TRANIG SET
traning_set = "data/inex05.train.elastic.tree"


#TEST SET
test_set = "data/inex05.test.elastic.tree"
test_set = dataset_parser(test_set)
random.shuffle(test_set)


# TRAIN CON VALIDATION
traning_set = dataset_parser_class(traning_set)
traning_set = divide_tre_validation_htm(traning_set)



htm , lamda = training_val (modello,hidden_state,m,lerning_rate,epoche,batch_size,traning_set[0][0],decay,stop_n,traning_set[0][1],batch_size,nome_file)


print("test...")

result 		= test(htm,lamda,test_set,m,hidden_state)


print("\n\n          FINAL  LOSS AND ACCURACY ON TEST           ",result)
print("\n\n")


#result = train_and_test(modello,hidden_state,m,lerning_rate,epoche,batch_size,traning_set[0])
'''

#SENZA VALIDAITON
traning_set = dataset_parser(traning_set)
random.shuffle(traning_set)

htm , lamda = training (modello,hidden_state,m,lerning_rate,epoche,batch_size,traning_set,decay,stop_n,batch_size)

'''
