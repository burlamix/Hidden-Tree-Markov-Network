import numpy as np
import pylab as pl

acc = np.loadtxt('plot_acc_3f_24c_b32_01') 
loss = np.loadtxt('plot_loss_3f_24c_b32_01') 

pl.plot(acc)
pl.plot(loss)
pl.show()