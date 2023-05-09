import os
import numpy as np
import random
import tensorflow as tf
from tensorflow import keras
from keras import utils
from keras import models
from keras import layers
import matplotlib.pyplot as plt

directory = os.getcwd() + '/dataset'
training_set, validation_set = utils.image_dataset_from_directory(
    directory,
    labels="inferred",
    label_mode="int",
    class_names=['real', 'generated'],
    color_mode="rgb",
    batch_size=32,
    image_size=(1024, 1024),
    shuffle=True,
    seed=1,
    validation_split=0.1,
    subset='both',
    interpolation="bilinear",
    follow_links=False,
    crop_to_aspect_ratio=False,
)

# plt.figure('dataset examples', figsize=(10, 10))
# for images, labels in validation_set.take(1):
#     for i in range(9):
#         ax = plt.subplot(3, 3, i+1)
#         plt.imshow(images[i].numpy().astype('uint8'))
#         plt.title(validation_set.class_names[int(labels[i])])
#         plt.axis('off')
# plt.show()

results = []

for i in range(1,11):

    network = models.Sequential()
    network.add(layers.Flatten(input_shape=(1024,1024,3)))
    for j in range(i):
        network.add(layers.Dense(16, activation='relu'))
    network.add(layers.Dense(1, activation='sigmoid'))

    network.compile(optimizer='rmsprop',
                    loss='binary_crossentropy',
                    metrics=['accuracy'])

    history = network.fit(training_set, epochs=5, batch_size=32, validation_data=validation_set)
    results.append(history.history)

acc = history.history['accuracy']
epochs=range(1, len(acc)+1)

plt.figure('Strata trenowania dla różnej ilości warstw')
for i in range(10):
    plt.plot(epochs, results[i]['loss'],label=f'ilość warstw ukrytych: {i+1}')
plt.xlabel('Epoki')
plt.ylabel('Strata')
plt.legend()
plt.show()

plt.figure('Strata walidacji dla różnej ilości warstw')
for i in range(10):
    plt.plot(epochs, results[i]['val_loss'],label=f'ilość warstw ukrytych: {i+1}')
plt.xlabel('Epoki')
plt.ylabel('Strata')
plt.legend()
plt.show()

plt.figure('Dokładność trenowania dla różnej ilości warstw')
for i in range(10):
    plt.plot(epochs, results[i]['accuracy'],label=f'ilość warstw ukrytych: {i+1}')
plt.xlabel('Epoki')
plt.ylabel('Dokładność')
plt.legend()
plt.show()

plt.figure('Dokładność walidacji dla różnej ilości warstw')
for i in range(10):
    plt.plot(epochs, results[i]['val_accuracy'],label=f'ilość warstw ukrytych: {i+1}')
plt.xlabel('Epoki')
plt.ylabel('Dokładność')
plt.legend()
plt.show()

