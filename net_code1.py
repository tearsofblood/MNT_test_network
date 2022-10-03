import numpy as np
import tensorflow as tf
import tensorflow.keras as K
relu = K.activations.relu
X = np.arange(-10, 10, dtype=float)

relu(X, threshold=4).numpy()
out = K.activations.sigmoid(X)
import matplotlib.pyplot as plt
plt.plot(out)
dense = K.layers.Dense(10)
out = dense(np.ones((1, 1000)))
plt.hist(dense.get_weights()[0].ravel())
K.regularizers.l1_l2()
input_shape = (1, 10, 3)
X = np.ones(input_shape)
conv1D = K.layers.Conv1D(filters=4, kernel_size=5, 
                         kernel_initializer=K.initializers.ones,
                         bias_initializer='zeros')
                         
conv1D.weights[0].shape

my_callbacks = [
    tf.keras.callbacks.EarlyStopping(patience=2, ),
    tf.keras.callbacks.ModelCheckpoint(filepath='model.{epoch:02d}-{val_loss:.2f}.h5'),
    tf.keras.callbacks.TensorBoard(log_dir='./logs'),
]

(X_train, y_train), (X_test, y_test) = tf.keras.datasets.cifar10.load_data()
y_train = K.utils.to_categorical(y_train, num_classes=10)
y_test = K.utils.to_categorical(y_test, num_classes=10)
model = K.Sequential()
model.add(K.Input(shape=(32,32,3)))
model.add(K.layers.Conv1D(filters=4, kernel_size=4, strides=(1)))
model.add(K.layers.Conv1D(filters=4, kernel_size=4, strides=(1)))
model.add(K.layers.Dense(10, activation='softmax'))

optimizer=tf.keras.optimizers.Adam(learning_rate=0.8e-3)

model.compile(optimizer=optimizer,
              loss='categorical_crossentropy',
              metrics=['accuracy'])
              
model.fit(X_train, y_train, validation_data=(X_test, y_test),
          epochs=10, callbacks=my_callbacks)
