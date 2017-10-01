import tensorflow as tf
import numpy as np

N_HIDDEN_STATES  = 4   #C
N_SYMBOLS = 3           #M
MAX_CHILD = 5         #L


def random_sum_one1(shape1):

    rand = tf.random_uniform([shape1], 0, 1, dtype=tf.float64)
    sum = tf.reduce_sum(rand, [0])

    rand_sum = tf.divide(rand, sum)

    return rand_sum

def random_sum_one2(axe,shape1,shape2):

    rand = tf.random_uniform([shape1, shape2], 0, 1, dtype=tf.float64)
    #calcolo la somma sull'asse dove il totale deve valere zero
    sum = tf.reduce_sum(rand, [axe])

    #nel caso l'asse non è lo zero lo espando duplico così da poter dividere la matrice random per esso
    if axe == 1:
        sum = tf.expand_dims(sum, 1)
        sum = tf.tile(sum, [1, shape2])


    rand_sum = tf.divide(rand, sum)

    return rand_sum

def random_sum_one3(axe,shape1,shape2,shape3=None):

    rand = tf.random_uniform([shape1, shape2, shape3], 0, 1, dtype=tf.float64)
    sum = tf.reduce_sum(rand, [axe])
    sum = tf.expand_dims(sum, 1)
    sum = tf.tile(sum, [1,shape2,1])
    rand_sum = tf.divide(rand, sum)

    return rand_sum



bi = random_sum_one2(0,N_HIDDEN_STATES,N_SYMBOLS)
pi = random_sum_one2(1,N_HIDDEN_STATES,MAX_CHILD)
sp_p = random_sum_one1(MAX_CHILD)
A = random_sum_one3(1,N_HIDDEN_STATES,N_HIDDEN_STATES,MAX_CHILD)

sess = tf.Session()

print(sess.run([pi,pi2]))