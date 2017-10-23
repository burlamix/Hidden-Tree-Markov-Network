import pylab as pl
import numpy as np
from io import StringIO

like_list = np.loadtxt('5like_list.out')
s1 = np.loadtxt('5s1.out')
s2 = np.loadtxt('5s2.out')
s3 = np.loadtxt('5s3.out')
s4 = np.loadtxt('5s4.out')


pl.plot(like_list)
pl.plot(s1)
pl.plot(s2)
pl.plot(s3)
pl.plot(s4)
pl.show()

