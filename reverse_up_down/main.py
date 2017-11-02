import numpy as np
import tensorflow as tf
from tre_simple import *
from parser import *
from parser_class import *
from E_M_utils import *

hidden_state = 10




#FILE1 = "inex05.train.elastic.tree"
#FILE2 = "inex05.test.elastic.tree"
FILE1 = "test_3.tree"
FILE2 = "test_2.tree"

epoche = 70

#data_set = dataset_parser_class(FILE1)

#val_test = divide_tre_validation(data_set)

data_test = dataset_parser(FILE2)

likelihood_test(data_test,epoche,hidden_state)


'''
pi_l,sp_p_l,A_l,bi_l=modello(val_test[0][0],epoche,6)
rate = testing(val_test[0][1],pi_l,sp_p_l,A_l,bi_l,6)
np.save('a/6_rate', rate) 

pi_l,sp_p_l,A_l,bi_l=modello(val_test[1][0],epoche,6)
rate = testing(val_test[1][1],pi_l,sp_p_l,A_l,bi_l,6)
np.save('b/6_rate', rate) 

pi_l,sp_p_l,A_l,bi_l=modello(val_test[2][0],epoche,6)
rate = testing(val_test[2][1],pi_l,sp_p_l,A_l,bi_l,6)
np.save('c/6_rate', rate) 



# hidden_state = 8
pi_l,sp_p_l,A_l,bi_l=modello(val_test[0][0],epoche,8)
rate = testing(val_test[0][1],pi_l,sp_p_l,A_l,bi_l,8)
np.save('a/8_rate', rate) 


pi_l,sp_p_l,A_l,bi_l=modello(val_test[1][0],epoche,8)
rate = testing(val_test[1][1],pi_l,sp_p_l,A_l,bi_l,8)
np.save('b/8_rate', rate) 

pi_l,sp_p_l,A_l,bi_l=modello(val_test[2][0],epoche,8)
rate = testing(val_test[2][1],pi_l,sp_p_l,A_l,bi_l,8)
np.save('c/8_rate', rate) 





# hidden_state = 10
pi_l,sp_p_l,A_l,bi_l=modello(val_test[0][0],epoche,10)
rate = testing(val_test[0][1],pi_l,sp_p_l,A_l,bi_l,10)
np.save('a/10_rate', rate) 

pi_l,sp_p_l,A_l,bi_l=modello(val_test[1][0],epoche,10)
rate = testing(val_test[1][1],pi_l,sp_p_l,A_l,bi_l,10)
np.save('b/10_rate', rate) 

pi_l,sp_p_l,A_l,bi_l=modello(val_test[2][0],epoche,10)
rate = testing(val_test[2][1],pi_l,sp_p_l,A_l,bi_l,10)
np.save('c/10_rate', rate) 



pi_l,sp_p_l,A_l,bi_l=modello(data_set,epoche,8)
rate = testing(data_test,pi_l,sp_p_l,A_l,bi_l,8)
np.save('test_rate', rate) 
''' 

#np.save('a/'+str(hidden_state)+'_rate', rate) 


#likelihood_test(data_set[8],epoche)
#likelihood_test(data_test,epoche,hidden_state)

#pi,sp_p,A,bi = likelihood(data_set,epoche)
