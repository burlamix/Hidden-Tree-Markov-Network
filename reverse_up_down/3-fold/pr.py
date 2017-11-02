import pylab as pl
import numpy as np
from io import StringIO



a6 = np.load('a6_rate.npy')
a8 = np.load('a8_rate.npy')
a10 = np.load('a10_rate.npy')

b6 = np.load('b6_rate.npy')
b8 = np.load('b8_rate.npy')
b10 = np.load('b10_rate.npy')

c6 = np.load('c6_rate.npy')
c8 = np.load('c8_rate.npy')
c10 = np.load('c10_rate.npy')


er6 = (a6+b6+c6)/3
er8 = (a8+b8+c8)/3
er10 = (a10+b10+c10)/3


print(er6 )
print(er8)
print(er10 )

t6=56551+915
t8=60525+1394
t10=62694+2220

scal =700

t6=t6/scal
t8=t8/scal
t10=t10/scal


tt=[t6,t8,t10]

#time = pl.plot(tt,label='tempo',marker='o', linestyle='--')

err = pl.plot([er6,er8,er10],label='Accuracy',marker='o')
pl.xlabel('C')
pl.ylabel('Accuracy')
pl.ylim([80,100])
pl.xticks([0,1,2],[6,8,10])
#pl.legend()
pl.show()

print("-")