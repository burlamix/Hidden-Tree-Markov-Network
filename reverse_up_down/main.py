import numpy as np

from tre_simple import *
from parser import *
from parser_class import *
from E_M_utils import *




#FILE2 = "inex05.train.elastic.tree"
#FILE2 = "inex05.test.elastic.tree"
#FILE1 = "test_2.tree"
#FILE2 = "test_2.tree"
#FILE1 = "train_10.tree"
FILE2 = "test_3.tree"

epoche = 15

#data_set = dataset_parser_class(FILE1)
data_test = dataset_parser(FILE2)


#pi_l,sp_p_l,A_l,bi_l=modello(data_set,epoche)#

#testing(data_test,pi_l,sp_p_l,A_l,bi_l)



likelihood_test(data_test,epoche)

#pi,sp_p,A,bi = likelihood(data_set,epoche)
