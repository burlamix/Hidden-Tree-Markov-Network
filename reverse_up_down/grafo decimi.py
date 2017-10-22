import pylab as pl
import numpy as numpy


x=14534634
x_list = []
for i in range(0,300):
	x=x/6
	x_list.append(x)

pl.plot(x_list)
pl.savefig('immagina.png')

