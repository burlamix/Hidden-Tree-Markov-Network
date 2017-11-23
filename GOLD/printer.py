import numpy as np
import pylab as pl

acc = np.loadtxt('loss_slo') 
loss = np.loadtxt('acc_slo') 

pl.plot(acc)
pl.plot(loss)
pl.show()
#pl.savefig('1_solo_traning.png')
