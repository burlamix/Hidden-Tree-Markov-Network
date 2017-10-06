import tensorflow as tf
import numpy as np
from datetime import datetime

-0.453

SIZE = 100

a = np.random.rand(SIZE,SIZE)
b = np.random.rand(SIZE,SIZE)

#ph_a = tf.placeholder(shape=[SIZE,SIZE], dtype=tf.float64)
#ph_b = tf.placeholder(shape=[SIZE,SIZE], dtype=tf.float64)
ph_a = tf.random_uniform([SIZE,SIZE], 0, 1, dtype=tf.float64)
ph_b = tf.random_uniform([SIZE,SIZE], 0, 1, dtype=tf.float64)

'''
start = datetime.now()
for i in range(0,1000):
    result = np.matmul(a,b)
    b=result
print("np  : ", (datetime.now() - start).total_seconds() * 1000)
'''

result_l = []

start = datetime.now()
for i in range(0,1000):
    result = tf.matmul(ph_a,ph_b)
    result_l.append(result)
    ph_b=result
print("for out       : ", (datetime.now() - start).total_seconds() * 1000)

start = datetime.now()
with tf.Session() as sess:

    sess.run(tf.global_variables_initializer(),)
    sess.run([result_l], {ph_b: b, ph_a: a})

print("run for out       : ", (datetime.now() - start).total_seconds() * 1000)



# con il for dento l a sesione

start = datetime.now()
for i in range(0,1000):
    result2 = tf.matmul(ph_a,ph_b)
print("for out       : ", (datetime.now() - start).total_seconds() * 1000)

start = datetime.now()
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer(), )

    for i in range(0, 1000):
        sess.run([result_l], {ph_b: b, ph_a: a})

print("run for out       : ", (datetime.now() - start).total_seconds() * 1000)

