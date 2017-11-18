import numpy as np
import pylab as pl

acc = np.loadtxt('plot_loss_soft_10_001') 
loss = np.loadtxt('plot_acc_soft_10_001') 

pl.plot(acc)
pl.plot(loss)
pl.show()
#pl.savefig('1_solo_traning.png')
