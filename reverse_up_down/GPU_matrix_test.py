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


rr=tf.zeros([3, 1], tf.int32)
for i in range(0,5):
    a = tf.slice(ph_1, [0, i], [3, 1])
    rr = tf.concat([rr,a],1)



input_1 = np.array([[1,2,3],
                   [4,5,6],
                   [7,8,9]])

#r = tf.slice(ph_1,[0,5], [3, 1])

print(ph_1)

rr= tf.tile(ph_1,[3,3,1])

with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print( sess.run(rr, feed_dict={ph_1: input_1}))
