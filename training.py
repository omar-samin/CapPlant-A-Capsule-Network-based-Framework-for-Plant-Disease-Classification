from keras.models import Sequential,Model
from keras import layers
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense,Dropout
from keras.utils import plot_model

import matplotlib.pyplot as plt
from keras.models import Sequential,model_from_json
from keras.preprocessing.image import ImageDataGenerator
import os
from keras.preprocessing.image import ImageDataGenerator
from capsule_layers import ConvCapsuleLayer, DeconvCapsuleLayer, Mask, Length

from keras.layers import Input, Conv2DTranspose
from keras.layers.advanced_activations import ReLU, LeakyReLU

PATH = os.path.join( 'gray_scale')

train_dir = os.path.join(PATH, 'train')
validation_dir = os.path.join(PATH, 'validation')
test_dir = os.path.join(PATH, 'test')
#total_train=43488
total_train=5000
total_val=5423
total_test=5470

print("--")
print("Total training images:", total_train)
print("Total validation images:", total_val)
print("Total validation images:", total_test)

batch_size = 32
epochs = 200
IMG_HEIGHT =224
IMG_WIDTH = 224


train_image_generator = ImageDataGenerator(rescale=1./255) # Generator for our training data
validation_image_generator = ImageDataGenerator(rescale=1./255) # Generator for our validation data
test_image_generator = ImageDataGenerator(rescale=1./255) # Generator for our testing data


train_data_gen = train_image_generator.flow_from_directory(batch_size=batch_size,
                                                           directory=train_dir,
                                                           shuffle=True,
                                                           target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                           class_mode='categorical')
val_data_gen = validation_image_generator.flow_from_directory(batch_size=batch_size,
                                                              directory=validation_dir,
                                                              target_size=(IMG_HEIGHT, IMG_WIDTH),
			                                      class_mode='categorical')
test_data_gen = test_image_generator.flow_from_directory(batch_size=batch_size,
                                                              directory=test_dir,
                                                              target_size=(IMG_HEIGHT, IMG_WIDTH),
			                                      class_mode='categorical')


input_shape = Input(shape=(IMG_HEIGHT,IMG_HEIGHT,3))
#Alexnet start
#Layer 1 
c1 = Conv2D(16, kernel_size=(3,3),strides=(1,1), activation='relu', padding='same')(input_shape)
c1 = MaxPooling2D(pool_size=(2,2),strides=(2,2))(c1)
#c1 = Dropout(0.2)(c1)
#Layer 2
c2 = Conv2D(32, kernel_size=(3,3), activation='relu', padding='same')(c1)
c2 =  MaxPooling2D(pool_size=(2,2),strides=(2,2))(c2)
#c2 = Dropout(0.2)(c2)

#Layer 3
c3= Conv2D(64, kernel_size=(3,3), activation='relu', padding='same')(c2)
c3 =  MaxPooling2D(pool_size=(2,2),strides=(2,2))(c3)
#c3 = Dropout(0.2)(c3)


#Layer 4
c4 = Conv2D(128, kernel_size=(3,3), activation='relu', padding='same')(c3)
c4 =  MaxPooling2D(pool_size=(2,2),strides=(2,2))(c4)
#c4 = Dropout(0.2)(c4)
'''
#Layer 5
c5 = Conv2D(256, kernel_size=(3,3), activation='relu', padding='same')(c4)
c5 = MaxPooling2D(pool_size=(2,2),strides=(2,2))(c5)
#c5 = Dropout(0.2)(c5)
'''
#capstart

#layer5
_, H, W, C = c4.get_shape()
c5 = layers.Reshape((H.value, W.value, 1, C.value))(c4)
c5 = ConvCapsuleLayer(kernel_size=3, num_capsule=1, num_atoms=256, strides=1, padding='same',routings=5, name='conv_cap')(c5)
_,A, H, W, C = c5.get_shape()
c5 = layers.Reshape((A.value, H.value, C.value))(c5)

#capend


c5 = Flatten()(c5)

#Layer 6
#c6 = Dense(256, activation='relu')(c5)

#Prediction
c7 = Dense(38, activation='softmax')(c5)

#Alexnet end
model = Model(inputs=[input_shape], outputs=[c7])

# Compiling the CNN
model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])


model.summary()
plot_model(model, to_file='Model/sketch6/model.png',show_shapes=True)
history = model.fit_generator(
    train_data_gen,
    steps_per_epoch=total_train // batch_size,
    epochs=epochs,
    validation_data=val_data_gen,
    validation_steps=total_val // batch_size
)
print(history.history.keys())
acc = history.history['acc']
val_acc = history.history['val_acc']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

# serialize model to JSON
model_json = model.to_json()
with open("Model/sketch6/model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("Model/sketch6/model.h5")
print("Saved model to disk")

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.savefig('Model/sketch6/plot')


json_file = open('Model/sketch6/model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json,custom_objects={'ConvCapsuleLayer': ConvCapsuleLayer})
# load weights into new model
loaded_model.load_weights("Model/sketch6/model.h5")
print("Loaded model from disk")


loaded_model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

#https://medium.com/@vijayabhaskar96/tutorial-image-classification-with-keras-flow-from-directory-and-generators-95f75ebe5720
STEP_SIZE_VALID=test_data_gen.n//test_data_gen.batch_size
test_loss, test_acc = loaded_model.evaluate_generator(test_data_gen,1)
print(test_acc)