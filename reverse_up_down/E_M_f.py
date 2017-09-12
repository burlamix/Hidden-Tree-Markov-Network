from E_M_utils import *
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

N_HIDDEN_STATES  = 4   #C
MAX_CHILD = 3          #L
N_TREE = 10             #NT
N_SYMBOLS = 5           #M
I_NODE = 70
MAX_LEVEL = 3

t = List_tree(1)
t.t.make_linear_tree(MAX_CHILD, MAX_LEVEL, N_SYMBOLS)
t.divide_leaves()
t.set_name()

#model parameters
#positional prior probability matrix --- pi
pi = np.ones((N_HIDDEN_STATES,MAX_CHILD))
#SP probability matrix --- fi
sp_p = np.ones((MAX_CHILD))
#positional state transiction probability matrix --- A
A = np.ones((N_HIDDEN_STATES,N_HIDDEN_STATES,MAX_CHILD)) # assumo che gli inidici sono j, i, l
#multinomial emision --- bi
bi = np.ones((N_HIDDEN_STATES,N_SYMBOLS))

#upward parameters beta
up_ward = np.ones((t.size,N_HIDDEN_STATES))
a_up_ward = np.ones((t.size,N_HIDDEN_STATES,MAX_CHILD))

#stater posterior €
post = np.ones((t.size,N_HIDDEN_STATES))
#pairwwise smoothed posterior
s_post = np.ones((MAX_CHILD,N_HIDDEN_STATES,N_HIDDEN_STATES,t.size))


#internal node prior
in_prior = np.zeros((N_HIDDEN_STATES,t.size))
in_prior2 = np.zeros((N_HIDDEN_STATES,t.size))


#pairwise smoothed posterior
E = np.zeros((N_HIDDEN_STATES,t.size))
EE = np.zeros((t.size,N_HIDDEN_STATES,N_HIDDEN_STATES))



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
w=1
for i in range(0,N_HIDDEN_STATES):
    for j in range(t.size-len(t.struct[-1]),t.size ):
        in_prior[i,j]=w
        w=w+1

for i in range(0,MAX_CHILD):
    sp_p[i]=i+1

w=0
for i in range(0,N_HIDDEN_STATES ):
    for j in range(0, N_HIDDEN_STATES):
        for k in range(0, MAX_CHILD):
            A[i,j,k]=w
            w=w+3
z=1
for i in range(0,t.size):
    for j in range(0, N_HIDDEN_STATES):
        up_ward[i,j]=z
        z=z+1


w=0
for i in range(0,t.size ):
    for j in range(0, N_HIDDEN_STATES):
        for k in range(0, MAX_CHILD):
            a_up_ward[i,j,k]=w
            w=w+1

z=0
for j in range(0, t.size ):
    for i in range(0,N_HIDDEN_STATES):
        E[i,j]=0
        z=z+1

ph_pi = tf.placeholder(shape=[N_HIDDEN_STATES,MAX_CHILD], dtype=tf.float64)
ph_sp_p = tf.placeholder(shape=[MAX_CHILD], dtype=tf.float64)
ph_A = tf.placeholder(shape=[N_HIDDEN_STATES,N_HIDDEN_STATES,MAX_CHILD], dtype=tf.float64)
ph_bi = tf.placeholder(shape=[N_HIDDEN_STATES,N_SYMBOLS], dtype=tf.float64)
ph_post = tf.placeholder(shape=[t.size,N_HIDDEN_STATES], dtype=tf.float64)
ph_s_post = tf.placeholder(shape=[MAX_CHILD,N_HIDDEN_STATES,N_HIDDEN_STATES,t.size], dtype=tf.float64)
ph_in_prior = tf.placeholder(shape=[N_HIDDEN_STATES,t.size], dtype=tf.float64)


init_prior = tf.constant(in_prior, dtype=tf.float64)
var_in_prior = tf.get_variable('var_in_prior', initializer=init_prior)

init_a_up_ward = tf.constant(a_up_ward, dtype=tf.float64)
var_a_up_ward = tf.get_variable('var_a_up_ward', initializer=init_a_up_ward)


init_up_ward = tf.constant(up_ward, dtype=tf.float64)
var_up_ward = tf.get_variable('var_up_ward', initializer=init_up_ward)


init_E = tf.constant(E, dtype=tf.float64)
var_E = tf.get_variable('E', initializer=init_E)

init_EE = tf.constant(EE, dtype=tf.float64)
var_EE = tf.get_variable('EE', initializer=init_EE)

#-------------------------------------------------------------------------------------------

(var_E,var_EE) = Reversed_Upward_Downward(var_E, var_EE, ph_sp_p, ph_A, ph_bi, ph_pi, var_in_prior, var_a_up_ward, var_up_ward, N_HIDDEN_STATES, MAX_CHILD, t)

#-------------------------------------------------------------------------------------------

with tf.Session() as sess:

    """writer = tf.summary.FileWriter("/home/simone/Documents/università/TESI/codici/reverse_up_down")
    writer.add_graph(sess.graph)"""

    sess.run(tf.global_variables_initializer(),)
    print(sess.run([var_EE,var_E], {ph_bi: bi, ph_pi: pi,ph_sp_p: sp_p, ph_A: A,ph_in_prior: in_prior, ph_A: A}))








