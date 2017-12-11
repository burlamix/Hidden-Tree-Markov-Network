#from keras_t_g import *
import random
from E_M_utils import *

#	PARAMETRI DEL MODELLO
np.set_printoptions(threshold=np.nan)



hidden_state = 8


epoche = 22


batch_size = 1

#TRANIG SET
#traning_set = "data/inex05.train.elastic.tree"

#TEST SET
#test_set = "data/inex05.test.elastic.tree"


traning_set = dataset_parser_class(traning_set)


test_set = dataset_parser(test_set)


pi,sp_p,A,bi= likelihood_test(test_set,epoche,hidden_state)

#pi_l,sp_p_l,A_l,bi_l=modello_3(traning_set,epoche,hidden_state)


rate = testing_3(test_set,pi_l,sp_p_l,A_l,bi_l,hidden_state)

print(rate)
