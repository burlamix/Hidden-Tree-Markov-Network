import numpy as np
import tensorflow as tf
from tre_simple import *
from parser import *
from parser_class import *
from E_M_utils import *

#FILE = "inex05.train.elastic.tree"
FILE = "test_40.tree"

epoche = 3

data_set = dataset_parser_class(FILE)
data_test = dataset_parser(FILE)


pi_l,sp_p_l,A_l,bi_l=modello(data_set,epoche)

testing(data_test,pi_l,sp_p_l,A_l,bi_l)



#likelihood_test(data_test,epoche)

#pi,sp_p,A,bi = likelihood(data_set,epoche)
