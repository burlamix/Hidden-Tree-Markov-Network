from E_M_utils import *
import sys

import numpy as np
import  time

print("leggo il dataset")
data = dataset_parser()

print("parto")
start = datetime.now()

pi,sp_p,A,bi,t_pi,t_sp_p,t_A,t_bi, tot = training(data,1)

print("TOTALE grafo : ",(datetime.now()-start).total_seconds() )

start = datetime.now()

with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    #print(sess.run([aux,  pi,t_pi,  sp_p,t_sp_p,  A,t_A,  bi,t_bi]))
    print(sess.run([pi,sp_p,A,bi,t_pi,t_sp_p,t_A,t_bi, tot]))


print("TOTALE esecuzione : ",(datetime.now()-start).total_seconds() )

'''

data = dataset_parser()

t=data[13]
N_HIDDEN_STATES=3
MAX_CHILD=33
N_SYMBOLS=366



#nel caso non vengano passati dei valori iniziali ai parametri essi venono inizializati random
pi = random_sum_one2(1, N_HIDDEN_STATES, MAX_CHILD)
sp_p = random_sum_one1(MAX_CHILD)
A = random_sum_one3(1, N_HIDDEN_STATES, N_HIDDEN_STATES, MAX_CHILD)
bi = random_sum_one2(0, N_HIDDEN_STATES, N_SYMBOLS)

scope_tree = "scope_n0"
var_in_prior_list = []

var_EE,var_E,var_in_prior = Reversed_Upward_Downward(sp_p, A, bi, pi, N_HIDDEN_STATES, MAX_CHILD,t)



with tf.Session() as sess:
    sess.run(tf.global_variables_initializer(),)

    for i in range(0,len(data)):
        sess.run([var_in_prior])
        var_in_prior_list.append(var_in_prior)

'''
