
import keras
from keras.models import Sequential
from keras.layers import Dense
import numpy as np

np.set_printoptions(threshold=np.nan)


model = Sequential()
model.add(Dense(32, activation='relu', input_dim=100))
model.add(Dense(10, activation='softmax'))
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])


# Generate dummy data
import numpy as np
data = np.random.random((1000, 100))
labels = np.random.randint(10, size=(1000, 1))

# Convert labels to categorical one-hot encoding
one_hot_labels = keras.utils.to_categorical(labels, num_classes=10)

print("---",model.get_weights())
# Train the model, iterating on the data in batches of 32 samples
model.fit(data, one_hot_labels, epochs=100, batch_size=32)

print("---",model.get_weights())
