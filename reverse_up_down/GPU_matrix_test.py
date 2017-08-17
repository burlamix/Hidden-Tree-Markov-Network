import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import tensorflow as tf
import numpy as np

ph_1 = tf.placeholder(shape=[3,3], dtype=tf.int32)
ph_2 = tf.placeholder(shape=[3,3], dtype=tf.int32)
ph_3 = tf.placeholder(shape=[2,3], dtype=tf.int32)
ph_4 = tf.placeholder(shape=[2,3], dtype=tf.int32)

x = []


#[0,2] o la riga, 2 la colonna(DI PARTENZA), [3,1] 3 quante righe ha la colonna, 1 quante colonne prendo
#tf.slice(ph_1,[0,3], [3, 1]



#for i in range(0,1):
#rr = tf.concat([rr[1,:,:],ph_1],2)



input_1 =          [[1,2,3],
                   [4,5,6],
                   [7,8,9]]

input_2 =          [[11,22,33],
                   [44,55,66],
                   [77,88,99]]
#r = tf.slice(ph_1,[0,5], [3, 1])
print(ph_1.shape)

a= tf.expand_dims(ph_1,0)
a= tf.tile(a,[3,1,1])
print(a.shape)

b= tf.expand_dims(ph_2,0)
b= tf.tile(b,[3,1,1])

#rr= tf.expand_dims(a,0)
rr = tf.stack([a,b],0)

with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print( sess.run([a,rr], feed_dict={ph_1: input_1,ph_2: input_2}),a.shape,rr.shape)
