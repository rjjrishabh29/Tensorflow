import tensorflow as tf
import numpy as np
from tensorflow import keras
def house_model(y_new):
    xs = np.array([1,2,3,4,5,6],dtype='float')
    ys = np.array([0.571,1.142,1.174,2.285,2.857,3.428],dtype='float')
    model = tf.keras.Sequential([keras.layers.Dense(units=1, input_shape=[1])])
    model.compile(optimizer='sgd', loss='mean_squared_error')
    model.fit(xs,ys,epochs=1000)
    return model.predict(y_new)[0]

prediction = house_model([7.0])
print(math.ceil(prediction))
