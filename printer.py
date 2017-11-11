import numpy as np
import pylab as pl

to_print = np.loadtxt('plot_acc_3_15_b32_01_inv') 

pl.plot(to_print)
pl.show()