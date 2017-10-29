
import edward as ed
import numpy as np
import tensorflow as tf

import matplotlib.pyplot as plt

from edward.models import Normal

CLASSI=11

def build_toy_dataset(N=40, noise_std=0.1):
  D = 1
  X = np.concatenate([np.linspace(0, 2, num=N / 2),
                      np.linspace(6, 8, num=N / 2)])
  y = np.cos(X) + np.random.normal(0, noise_std, size=N)
  X = (X - 4.0) / 4.0
  X = X.reshape((N, D))
  return X, y


def neural_network(X):
  h = tf.tanh(tf.matmul(X, W_0) )
  h = tf.nn.softmax(tf.matmul(h, W_1))
  return tf.reshape(h, [-1])


ed.set_seed(42)

N = 40  # number of data points
D = 1   # number of features

# DATA
X_train, y_train = build_toy_dataset(N)

# MODEL
W_0 = np.ones([D, CLASSI],dtype=np.float32)
W_1 = Normal(loc=tf.zeros([CLASSI, 1]), scale=tf.ones([CLASSI, 1]))

X = tf.placeholder(tf.float32, [N, D])
y = Normal(loc=neural_network(X), scale=0.1 * tf.ones(N))

# INFERENCE
qW_1 = Normal(loc=tf.Variable(tf.random_normal([CLASSI, 1])),
              scale=tf.nn.softplus(tf.Variable(tf.random_normal([CLASSI, 1]))))

inference = ed.KLqp({W_1: qW_1}, data={X: X_train, y: y_train})
inference.run(n_iter=10)