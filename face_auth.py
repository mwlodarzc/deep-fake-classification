import os
import numpy as np
import random
import tensorflow as tf
from tensorflow import keras
from keras import utils
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

plt.figure('dataset examples', figsize=(10, 10))
for images, labels in validation_set.take(1):
    for i in range(9):
        ax = plt.subplot(3, 3, i+1)
        plt.imshow(images[i].numpy().astype('uint8'))
        plt.title(validation_set.class_names[int(labels[i])])
        plt.axis('off')
plt.show()
