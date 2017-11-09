#from keras_t_g import *
import random
from E_M_utils import *

#	PARAMETRI DEL MODELLO

m=30
hidden_state = 3


epoche = 35


batch_size = 1

#TRANIG SET
#traning_set = "data/inex05.train.elastic.tree"
#traning_set = "data/train_1000.tree"
traning_set = "data/test_3.tree"

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
