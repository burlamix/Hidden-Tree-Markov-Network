from E_M_utils import *
from parser import *

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

N_HIDDEN_STATES  = 4   #C

N_SYMBOLS = 366           #M


"""
#serve solo per la costruzione dell'albero lineare
MAX_LEVEL = 3
MAX_CHILD = 3          #L
t = Tree(1,1)
t.t.make_linear_tree(MAX_CHILD, MAX_LEVEL, N_SYMBOLS)
t.divide_leaves()
t.set_name()
"""
MAX_CHILD = 30          #L

t = dataset_parser()


#model parameters
#positional prior probability matrix --- pi
pi = np.ones((N_HIDDEN_STATES,MAX_CHILD))
#SP probability matrix --- fi
sp_p = np.ones((MAX_CHILD))
#positional state transiction probability matrix --- A
A = np.ones((N_HIDDEN_STATES,N_HIDDEN_STATES,MAX_CHILD)) # assumo che gli inidici sono j, i, l
#multinomial emision --- bi
bi = np.ones((N_HIDDEN_STATES,N_SYMBOLS))



################################################################################ tutti questi vanno dentro la funzio




#___row test______________
z=100
h=0
for i in range(0,N_HIDDEN_STATES):
    for j in range(0, N_SYMBOLS):
        bi[i,j]=z
        h=h+1
        z=z+1
for i in range(0,N_HIDDEN_STATES ):
    for j in range(0, MAX_CHILD):
        pi[i,j]=h
        h=h+1


for i in range(0,MAX_CHILD):
    sp_p[i]=i+1

w=0
for i in range(0,N_HIDDEN_STATES ):
    for j in range(0, N_HIDDEN_STATES):
        for k in range(0, MAX_CHILD):
            A[i,j,k]=w
            w=w+3


ph_pi = tf.placeholder(shape=[N_HIDDEN_STATES,MAX_CHILD], dtype=tf.float64)
ph_sp_p = tf.placeholder(shape=[MAX_CHILD], dtype=tf.float64)
ph_A = tf.placeholder(shape=[N_HIDDEN_STATES,N_HIDDEN_STATES,MAX_CHILD], dtype=tf.float64)
ph_bi = tf.placeholder(shape=[N_HIDDEN_STATES,N_SYMBOLS], dtype=tf.float64)





#----------qualcosa da definire---------------------------------------------------------------------------------



#-------------------------------------------------------------------------------------------

#for k in range(i,2000):
#with tf.variable_scope("conv1"):
(var_EE) = Reversed_Upward_Downward(ph_sp_p, ph_A, ph_bi, ph_pi, N_HIDDEN_STATES, MAX_CHILD, t[k])

#-------------------------------------------------------------------------------------------

with tf.Session() as sess:

    """writer = tf.summary.FileWriter("/home/simone/Documents/università/TESI/codici/reverse_up_down")
    writer.add_graph(sess.graph)"""

    sess.run(tf.global_variables_initializer(),)
    print(sess.run([var_EE], {ph_bi: bi, ph_pi: pi,ph_sp_p: sp_p, ph_A: A}))








