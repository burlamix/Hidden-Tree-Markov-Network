from E_M_utils import *
import numpy as np
import  time
data= dataset_parser()

training(data,1)
'''
SIZE = 15
TREE=15

A = np.ones((SIZE,SIZE,SIZE)) # assumo che gli inidici sono j, i, l
B = np.ones((SIZE,SIZE,SIZE)) # assumo che gli inidici sono j, i, l
up_ward = np.ones((SIZE, SIZE))

z = 1
for i in range(0, SIZE):
    for j in range(0, SIZE):
        up_ward[i, j] = z
        z = z + 1


w=0
for i in range(0,SIZE ):
    for j in range(0, SIZE):
        for k in range(0, SIZE):
            A[i,j,k]=w
            B[i,j,k]=w+1
            w=w+1

init_up_ward = tf.constant(up_ward, dtype=tf.float64)
var_up_ward = tf.get_variable('var_up_ward', initializer=init_up_ward)
ph_A = tf.placeholder(shape=[SIZE,SIZE,SIZE], dtype=tf.float64)
ph_B = tf.placeholder(shape=[SIZE,SIZE,SIZE], dtype=tf.float64)
print("_______________")


start33 = datetime.now()
a_sli_E = tf.slice(up_ward, [4,0], [2,SIZE])
print(a_sli_E)
print("slice :               ", (datetime.now() - start33).total_seconds() * 1000)


start33 = datetime.now()
a = tf.gather(up_ward,  [1,2,3,4,5,6,7,8,9,10,3,6,7,7,5,2,2,3,4,5,3,5,7,8,7,5,4,3,2,1,2,4,5,6,7,6,5,4,5,4,5,6,7,8,7] )
print(a)
print("gather :               ", (datetime.now() - start33).total_seconds() * 1000)


with tf.Session() as sess:

    """writer = tf.summary.FileWriter("/home/simone/Documents/università/TESI/codici/reverse_up_down")
    writer.add_graph(sess.graph)"""

    sess.run(tf.global_variables_initializer(),)
    print(sess.run([var_up_ward,a_sli_E,a], {ph_A: A}))

'''
