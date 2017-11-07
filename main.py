from keras_t_g import *
import random

#	PARAMETRI DEL MODELLO

m=30
hidden_state = 10

lerning_rate=0.01
decay=1e-2

epoche = 35


batch_size = 32

#TRANIG SET
traning_set = "data/inex05.train.elastic.tree"
#traning_set = "data/train_1000.tree"
#traning_set = "data/test_1.tree"

#TEST SET
test_set = "data/inex05.test.elastic.tree"
#test_set = "data/test_1000.tree"
#test_set = "data/test_1.tree"


#Train senza validation
data_train = dataset_parser(traning_set)
random.shuffle(data_train)

data_test = dataset_parser(test_set)

#train con 3-validation
#data_train = dataset_parser_class(traning_set)
#data_train = divide_tre_validation_htm(data_train)


modello = HTM(m,lerning_rate,decay)




#result = train_and_test(modello,hidden_state,m,lerning_rate,epoche,batch_size,data_train[0])

htm , lamda = training(modello,hidden_state,m,lerning_rate,epoche,batch_size,data_train,decay)

print("test...")

result 		= test(htm,lamda,data_test,m,hidden_state)





print("\n\n          FINAL  LOSS AND ACCURACY ON TEST           ",result)
print("\n\n")




''' 
	iterable = [data_train[0],data_train[1],data_train[2]]
	pool = multiprocessing.Pool()
	func = partial(train_and_test, modello,hidden_state,m,lerning_rate,epoche,batch_size)
	result = pool.map(func, iterable)
	pool.close()
	pool.join()
	print(result)
'''