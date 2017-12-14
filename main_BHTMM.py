import random
from BHTMM import *
np.set_printoptions(threshold=np.nan)

#	PARAMETRI DEL MODELLO

hidden_state = 8
epoche = 22

batch_size = 1

#TRANIG SET
traning_set = "data/inex06.train.elastic.tree"

#TEST SET
test_set = "data/inex06.test.elastic.tree"

print("----------")
traning_set = dataset_parser_class(traning_set)

test_set = dataset_parser(test_set)

pi,sp_p,A,bi= likelihood_test(test_set,epoche,hidden_state)


rate = testing_3(test_set,pi_l,sp_p_l,A_l,bi_l,hidden_state)

print(rate)
