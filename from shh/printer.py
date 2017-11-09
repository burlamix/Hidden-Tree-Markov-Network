import numpy as np
import pylab as pl

to_print = np.loadtxt('plot_tti_01_e2') 

pl.plot(to_print)
pl.show()