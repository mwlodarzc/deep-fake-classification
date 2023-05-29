import os
import numpy as np


import tensorflow as tf
from tensorflow import keras
from tensorflow.python.keras import backend
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
import matplotlib.pyplot as plt

from cv2 import imread, resize, imshow, cvtColor,COLOR_BGR2GRAY
# gpu_options = tf.GPUOptions(allow_growth=True)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(log_device_placement=True))
backend.set_session(sess)


# directory = os.getcwd() + '/dataset'
# training_set, validation_set = utils.image_dataset_from_directory(
#     directory,
#     labels="inferred",
#     label_mode="int",
#     class_names=['real', 'generated'],
#     color_mode="rgb",
#     batch_size=32,
#     image_size=(1024, 1024),
#     shuffle=True,
#     seed=1,
#     validation_split=0.1,
#     subset='both',
#     interpolation="bilinear",
#     follow_links=False,
#     crop_to_aspect_ratio=False,
# )

# ABS_PATH='.'
ABS_PATH = './ffhq_mix'


class FFHQ_mixed(keras.utils.Sequence):
  
  def __init__(self, image_filenames, labels, batch_size) :
    self.image_filenames = image_filenames
    self.labels = labels
    self.batch_size = batch_size
    
  def __len__(self) :
    return (np.ceil(len(self.image_filenames) / float(self.batch_size))).astype(np.int)
  
  def __getitem__(self, idx) :
    batch_x = self.image_filenames[idx * self.batch_size : (idx+1) * self.batch_size]
    batch_y = self.labels[idx * self.batch_size : (idx+1) * self.batch_size]
    return np.array([(imread(f'{ABS_PATH}/restructured/{str(file_name)}'))/255.0 for file_name in batch_x]), np.array(batch_y)


batch_size = 20

train_generator = FFHQ_mixed(np.load(f'{ABS_PATH}/X_train_filenames.npy'), np.load(f'{ABS_PATH}/y_train.npy'), batch_size)
val_generator = FFHQ_mixed(np.load(f'{ABS_PATH}/X_val_filenames.npy'), np.load(f'{ABS_PATH}/y_val.npy'), batch_size)

# image_arr, label_arr = train_generator.__getitem__(1)
# print(np.shape(image_arr))
# print(np.shape(label_arr))



# plt.imshow(train_generator.__getitem__(1)[0][0])
# plt.show()
# plt.figure('dataset examples', figsize=(10, 10))
# for images, labels in validation_set.take(1):
#     for i in range(9):
#         ax = plt.subplot(3, 3, i+1)
#         plt.imshow(images[i].numpy().astype('uint8'))
#         plt.title(validation_set.class_names[int(labels[i])])
#         plt.axis('off')
# plt.show()


callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=6)

network = Sequential()
# network.add(layers.Flatten(input_shape=(512,512)))
network.add(Conv2D(16, (2,2),strides=(2,2), activation='relu', input_shape=(1024,1024,3))) # 512x512x16  
network.add(Conv2D(32, (10,10),strides=(2,2), activation='relu')) # 252x252x32  
network.add(MaxPooling2D(pool_size=(2, 2),strides=(2,2))) # 126x126x32
network.add(Conv2D(64, (6,6), strides=(2,2) ,activation='relu')) # 61x61x64
network.add(MaxPooling2D(pool_size=(3, 3),strides=(2,2))) # 30x30x64
network.add(Conv2D(128, (3, 3),padding='same', activation='relu')) #30x30x128
network.add(Conv2D(256, (3, 3), activation='relu')) #28x28x256
network.add(Conv2D(256, (2, 2),strides=(2,2), activation='relu')) #14x14x512
network.add(MaxPooling2D(pool_size=(2, 2),strides=(2,2))) # 6x6x512
network.add(Flatten())
network.add(Dense(1000, activation='relu'))
network.add(Dense(500, activation='relu'))
network.add(Dense(500, activation='relu'))
network.add(Dense(100, activation='relu'))
network.add(Dense(1, activation='sigmoid'))

network.compile(optimizer='rmsprop',
                    loss='binary_crossentropy',
                    metrics=['accuracy'])

history = network.fit_generator(
  generator = train_generator,
  steps_per_epoch=20,
  epochs=300,
  # callbacks=[callback],
  validation_steps=10,
  validation_data=val_generator
)
network.save(f'.\{ABS_PATH}\\ffhq_mix_p')
# print(np.shape(history.history))

print(f"Training accuracy: {history.history['accuracy']}")
print(f"Training loss: {history.history['loss']}")
print(f"Validation accuracy: {history.history['val_accuracy']}")
print(f"Validation loss: {history.history['val_loss']}")

plt.figure(1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()

plt.figure(2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()