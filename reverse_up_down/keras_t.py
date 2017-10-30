import keras
from keras.models import Sequential
from keras.layers import Input, Dense
from keras.models import Model
import numpy as np
import scipy.misc as sc


M=20
K=11



cl_size = sc.comb(M, 2).astype(np.int64)


model = Sequential()
model.add(Dense(cl_size, activation='tanh',trainable=False,kernel_initializer=keras.initializers.Ones(), input_dim=M))
model.add(Dense(K, activation='softmax'))
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Generate dummy data
import numpy as np
data = np.random.random((1000, M))
labels = np.random.randint(K, size=(1000, 1))

# Convert labels to categorical one-hot encoding
one_hot_labels = keras.utils.to_categorical(labels, num_classes=K)

# Train the model, iterating on the data in batches of 32 samples
model.fit(data, one_hot_labels, epochs=100, batch_size=32)