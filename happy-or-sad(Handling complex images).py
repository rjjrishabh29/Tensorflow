

# Below is code with a link to a happy or sad dataset which contains 80 images, 40 happy and 40 sad. 
# Create a convolutional neural network that trains to 100% accuracy on these images,  which cancels training upon hitting training accuracy of >.999
# 
# Hint -- it will work best with 3 convolutional layers.




import tensorflow as tf
import os
import zipfile
from os import path, getcwd, chdir

!wget --no-check-certificate \
    "https://storage.googleapis.com/laurencemoroney-blog.appspot.com/happy-or-sad.zip" \
    -O "/tmp/happy-or-sad.zip"

zip_ref = zipfile.ZipFile("/tmp/happy-or-sad.zip", 'r')
zip_ref.extractall("/tmp/h-or-s")
zip_ref.close()


# Directory with our happy pictures
happy_dir = os.path.join('/tmp/h-or-s/happy')

# Directory with our sad pictures
sad_dir = os.path.join('/tmp/h-or-s/sad')

train_happy = os.listdir(happy_dir)
#print(train_happy[:10])

train_sad = os.listdir(sad_dir)
#print(train_sad[:10])

#print('total training happy images:', len(os.listdir(happy_dir)))
#print('total training sad images:', len(os.listdir(sad_dir)))






def train_happy_sad_model():

    DESIRED_ACCURACY = 0.999

    class myCallback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs={}):
            if(logs.get('acc')>DESIRED_ACCURACY):
                print("\nReached 99% accuracy so cancelling training!")
                self.model.stop_training = True


    callbacks = myCallback()
    
    model = tf.keras.models.Sequential([
    # This is the first convolution
    tf.keras.layers.Conv2D(16, (3,3), activation='relu', input_shape=(200, 200, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    # The second convolution
    tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    # The third convolution
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    # Flatten the results to feed into a DNN
    tf.keras.layers.Flatten(),
    # 256 neuron hidden layer
    tf.keras.layers.Dense(256, activation='relu'),
    # 512 neuron hidden layer
    tf.keras.layers.Dense(512, activation='relu'),
    # Only 1 output neuron. It will contain a value from 0-1 where 0 for 1 class ('happy') and 1 for the other ('sad')
    tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    from tensorflow.keras.optimizers import RMSprop

    model.compile(loss='binary_crossentropy',
              optimizer=RMSprop(lr=0.001),
              metrics=['accuracy'])

    from tensorflow.keras.preprocessing.image import ImageDataGenerator

    train_datagen = ImageDataGenerator(rescale=1/255)

    # a target_size of 150 X 150.
    train_generator = train_datagen.flow_from_directory('/tmp/h-or-s/',  # This is the source directory for training images
        target_size=(200, 200),  # All images will be resized to 150x150
        batch_size=15, # Since we use binary_crossentropy loss, we need binary labels
        class_mode='binary')

    
    history = model.fit_generator(
           train_generator,
      steps_per_epoch=16,  
      epochs=15,
      verbose=1,callbacks=[callbacks])
    # model fitting
    return history.history['acc'][-1]

train_happy_sad_model()



