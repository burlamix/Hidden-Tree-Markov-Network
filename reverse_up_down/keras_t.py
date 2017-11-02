import keras
from keras.models import Sequential
from keras.layers import Input, Dense
from keras.models import Model
import numpy as np
import scipy.misc as sc
from keras import backend as bk

M=4
K=11

cl_size = sc.comb(M, 2).astype(np.int64)

def my_init2(shape, dtype=None):
	m_init = np.zeros(shape, dtype=dtype)
	p=0
	s=1
	for i in range(0,shape[1]):
		m_init[p,i]=1
		m_init[s,i]=-1
		if(s==shape[0]-1):
			p=p+1
			s=p
		s=s+1
	return m_init








model = Sequential()
model.add(Dense(cl_size, activation='tanh',trainable=False,kernel_initializer=my_init2, input_dim=M))
model.add(Dense(K, activation='softmax'))
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Generate dummy data
data = np.random.random((10, M))
labels = np.random.randint(K, size=(10, 1))
one_hot_labels = keras.utils.to_categorical(labels, num_classes=K)
print(data)
print(labels)
print(one_hot_labels)



for i in range (0,1000):
	res = model.predict( data)
	model.fit(data, one_hot_labels, epochs=1, batch_size=32)
	


#model.fit(data, one_hot_labels, epochs=1000, batch_size=32)




# Train the model, iterating on the data in batches of 32 samples


