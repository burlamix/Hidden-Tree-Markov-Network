import numpy as np
import tensorflow as tf
import tre_simple as ts
MAX_CHILD=3
DAT=10



N_HIDDEN_STATES =3


epoche =2


#per il numero delle epoco eseguo l'E-M
prima_epoca = None
for zzzz in range(0, epoche):
    print("EPOCA: ",zzzz)

    scope_tree = "scope_n0"

    for j in range(0,10):
        scope_tree=scope_tree[:-len(str(j-1))]+str(j)
        with tf.variable_scope(scope_tree,reuse=prima_epoca):

            in_prior = np.ones((N_HIDDEN_STATES, N_HIDDEN_STATES))

            for ii in range(0, N_HIDDEN_STATES):
                    #for jj in range(t.size - len(t.struct[-1]), t.size):
                for jj in range(N_HIDDEN_STATES):
                    in_prior[ii, jj] = ii

            init_prior = tf.constant(in_prior, dtype=tf.float64)
            var_in_prior = tf.get_variable('var_in_prior', initializer=init_prior)

