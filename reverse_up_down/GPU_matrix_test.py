import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import tensorflow as tf
import numpy as np
N_HIDDEN_STATES  = 4   #C
MAX_CHILD = 3          #L
N_TREE = 10             #NT
N_SYMBOLS = 9           #M
N_NODE = 100            #N
I_NODE = 70
MAX_LEVEL = 3

ph_1 = tf.placeholder(shape=[3,3], dtype=tf.int32)
ph_2 = tf.placeholder(shape=[3,4], dtype=tf.int32)
ph_3 = tf.placeholder(shape=[2,3], dtype=tf.int32)
ph_4 = tf.placeholder(shape=[2,3], dtype=tf.int32)
#var = tf.Variable(tf.ones([3,4], dtype=tf.int32), dtype=tf.int32)



#[0,2] o la riga, 2 la colonna(DI PARTENZA), [3,1] 3 quante righe ha la colonna, 1 quante colonne prendo
#tf.slice(ph_1,[0,3], [3, 1]

input_1 =          [[1,2,3],
                   [4,5,6],
                   [7,8,9]]

input_2 =          [[11,22,33,44],
                   [44,55,66,55],
                   [77,88,99,999]]

init = tf.constant(input_2)
var = tf.get_variable('var', initializer=init)


#r = tf.slice(ph_1,[0,5], [3, 1])

#a= tf.expand_dims(ph_1,0)
#a= tf.tile(a,[3,1,1])

b= tf.expand_dims(ph_2,0)
b= tf.tile(b,[3,1,1])

s1=tf.reduce_sum(b,[1,2])
s2=tf.reduce_sum(b,[1,2])

s3 = tf.stack([s1,s2],0)  # concateno tutte le matrici N_HIDDEN_STATES*L dei node_prior per ogni gruppo di figli di un nodo  in un unica matrice N_HIDDEN_STATES*L*(numero di nodi del livello)

#rr= tf.expand_dims(a,0)
#rr = tf.stack([a,b],0)

s3= tf.transpose(s3)

#tf.scatter_update(var)

rr = tf.slice(ph_2, [0, 1], [3, 2])  # estraggo il node prior

ph_A = tf.placeholder(shape=[MAX_CHILD,N_HIDDEN_STATES,N_HIDDEN_STATES,N_SYMBOLS], dtype=tf.int32)

A = np.ones((MAX_CHILD, N_HIDDEN_STATES,N_HIDDEN_STATES,N_SYMBOLS))

w=0
for i in range(0,MAX_CHILD ):
    for j in range(0, N_HIDDEN_STATES):
        for k in range(0, N_HIDDEN_STATES):
            for z in range(0, N_SYMBOLS):
                A[i,j,k,z]=w
                w=w+1
print(A)

rr = tf.reduce_sum(ph_A, [3])  # sommo le matrici di dim 2


with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print( sess.run([rr], feed_dict={ph_A: A,ph_2: input_2}))
