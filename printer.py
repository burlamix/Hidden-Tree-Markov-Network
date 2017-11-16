import numpy as np
import pylab as pl

to_print = np.loadtxt('plot_loss_14s_o_tt_b1_ep25_lr_00005') 
to_print2 = np.loadtxt('plot_acc_14s_o_tt_b1_ep25_lr_00005') 

pl.plot(to_print)
pl.plot(to_print2)
pl.show()