import numpy as np
import scipy.misc as sc

#prima su internet presa a caso
def softmax(X, theta=1.0, axis=None):
    """
    Compute the softmax of each element along an axis of X.

    Parameters
    ----------
    X: ND-Array. Probably should be floats.
    theta (optional): float parameter, used as a multiplier
        prior to exponentiation. Default = 1.0
    axis (optional): axis to compute values along. Default is the
        first non-singleton axis.

    Returns an array the same size as X. The result will sum to 1
    along the specified axis.
    """

    # make X at least 2d
    y = np.atleast_2d(X)

    # find axis
    if axis is None:
        axis = next(j[0] for j in enumerate(y.shape) if j[1] > 1)

    # multiply y against the theta parameter,
    y = y * float(theta)

    # subtract the max for numerical stability
    y = y - np.expand_dims(np.max(y, axis=axis), axis)

    # exponentiate y
    y = np.exp(y)

    # take the sum along the specified axis
    ax_sum = np.expand_dims(np.sum(y, axis=axis), axis)

    # finally: divide elementwise
    p = y / ax_sum

    # flatten if X was 1D
    if len(X.shape) == 1: p = p.flatten()

    return p

N_tree = 5
N_classes = 7
hiddenLayerSize = sc.comb(N_tree, 2).astype(np.int64)

W_f = np.ones((N_tree, hiddenLayerSize))
W_0 = np.random.randn(hiddenLayerSize, N_classes)
x = np.random.randn(1,N_tree)


def neural_network(X):
  h = np.tanh(np.matmul(X, W_f) )
  h = softmax(np.matmul(h, W_0))
  return h

h=neural_network(x)

print (h)

