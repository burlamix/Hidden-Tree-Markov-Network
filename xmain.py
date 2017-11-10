#from keras_t_g import *
import random
from E_M_utils import *

#	PARAMETRI DEL MODELLO



hidden_state = 11


epoche = 6


batch_size = 1

#TRANIG SET
#traning_set = "data/inex05.train.elastic.tree"
traning_set = "data/train_666.tree"
#traning_set = "data/test_3.tree"

#TEST SET
#test_set = "data/inex05.test.elastic.tree"
test_set = "data/test_666.tree"
#test_set = "data/test_4.tree"


traning_set = dataset_parser_class(traning_set)



test_set = dataset_parser(test_set)



print("traning")
#pi,sp_p,A,bi= training(test_set,epoche,hidden_state)

pi_l,sp_p_l,A_l,bi_l=modello_3(traning_set,epoche,hidden_state)

#likelihood_test(test_set,epoche,hidden_state)


print("testin")
rate = testing_3(test_set,pi_l,sp_p_l,A_l,bi_l,hidden_state)

#print(rate)




#data_test = dataset_parser(test_set)
#random.shuffle(data_test)

'''
m=30
hidden_state = 3


epoche = 40


batch_size = 1

#TRANIG SET
#traning_set = "data/inex05.train.elastic.tree"
#traning_set = "data/train_1000.tree"
traning_set = "data/test_1.tree"

#TEST SET
#test_set = "data/inex05.test.elastic.tree"
#test_set = "data/test_1000.tree"
#test_set = "data/test_4.tree"


#Train senza validation
data_train = dataset_parser(traning_set)
random.shuffle(data_train)

#data_test = dataset_parser(test_set)
#random.shuffle(data_test)

likelihood_test(data_train,epoche,hidden_state)
''' 