import numpy as np
import tensorflow as tf
from tre_simple import *
from parser import *
from parser_class import *
from E_M_utils import *

hidden_state = 2


FILE1 = "p12"
#FILE1 = "p13"
#FILE1 = "p23"
#FILE2 = "p1"
#FILE3 = "p2"
FILE4 = "p3"

#FILE1 = "inex05.train.elastic.tree"
#FILE2 = "inex05.test.elastic.tree"
#FILE1 = "test_40.tree"
#FILE2 = "test_40.tree"

epoche = 25

data_set = dataset_parser_class(FILE1)
data_test = dataset_parser(FILE2)


pi_l,sp_p_l,A_l,bi_l=modello(data_set,epoche,hidden_state)

np.save('a/'+str(hidden_state)+'_pi_l', pi_l) 
np.save('a/'+str(hidden_state)+'_sp_p_l', sp_p_l) 
np.save('a/'+str(hidden_state)+'_A_l', A_l) 
np.save('a/'+str(hidden_state)+'_bi_l', bi_l) 


rate = testing(data_test,pi_l,sp_p_l,A_l,bi_l,hidden_state)

np.save('a/'+str(hidden_state)+'_rate.out', rate) 


#likelihood_test(data_set[8],epoche)
#likelihood_test(data_test,epoche,hidden_state)

#pi,sp_p,A,bi = likelihood(data_set,epoche)
