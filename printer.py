import numpy as np
import pylab as pl

to_print = np.loadtxt('plot_quato_inex') 

pl.plot(to_print)
pl.show()