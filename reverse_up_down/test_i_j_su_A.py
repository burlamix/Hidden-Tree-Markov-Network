from E_M_utils import *
from parser import *

N_HIDDEN_STATES1  = 3   #C
N_HIDDEN_STATES2  = 4   #C

N_SYMBOLS = 366           #M

MAX_CHILD = 2          #L


A = np.ones((N_HIDDEN_STATES1,N_HIDDEN_STATES2,MAX_CHILD)) # assumo che gli inidici sono j, i, l
sp_p = np.ones(MAX_CHILD)

w=0
for i in range(0,N_HIDDEN_STATES1 ):
    for j in range(0, N_HIDDEN_STATES2):
        for k in range(0, MAX_CHILD):
            A[i,j,k]=w
            w=w+1


ph_A = tf.placeholder(shape=[N_HIDDEN_STATES1, N_HIDDEN_STATES2, MAX_CHILD], dtype=tf.float64)
ph_sp_p = tf.placeholder(shape=[MAX_CHILD], dtype=tf.float64)

aux1 = tf.multiply(ph_sp_p, ph_A)  # broadcast implicito

#(var_EE) = Reversed_Upward_Downward(var_E, var_EE, ph_sp_p, ph_A, ph_bi, ph_pi, var_in_prior, var_a_up_ward, var_up_ward, N_HIDDEN_STATES, MAX_CHILD, t)




with tf.Session() as sess:

    """writer = tf.summary.FileWriter("/home/simone/Documents/università/TESI/codici/reverse_up_down")
    writer.add_graph(sess.graph)"""

    sess.run(tf.global_variables_initializer(),)
    print(sess.run([ph_A], {ph_A: A, ph_sp_p: sp_p}))

