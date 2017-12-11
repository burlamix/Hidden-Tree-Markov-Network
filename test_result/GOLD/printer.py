import numpy as np
import pylab as pl

loss1 = np.loadtxt('loss_slo') 
acc1 = np.loadtxt('acc_slo') 


loss2 = np.loadtxt('loss_fast') 
acc2 = np.loadtxt('acc_fast') 


#pl.plot(acc1)
pl.plot(acc2)

#pl.plot(loss1,linestyle='--')
pl.plot(loss2,linestyle='--')
pl.show()
#pl.savefig('1_solo_traning.png')
