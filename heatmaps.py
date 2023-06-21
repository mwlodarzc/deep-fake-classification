import os
import numpy as np
import math
import random
import tensorflow as tf
from tensorflow import keras
from keras import utils
from keras import models
from keras import layers
import matplotlib.pyplot as plt
from PIL import Image
import cv2
from keras import Model

network = tf.keras.saving.load_model('models/maximal_99.5p_net/ffhq_mix_p/')

#example image loading
image_path = 'heatmaps/test_image.jpeg'  
image = Image.open(image_path)
test_image = np.array(image)
plt.imshow(test_image)
plt.axis('off')
plt.show()

layer_number = 9

#all chanels heatmaps
for i in range(layer_number) :
    activation_model = Model(inputs=network.input, outputs=network.layers[i].output)

    activations = activation_model.predict(np.expand_dims(test_image, axis=0))

    num_channels = activations.shape[-1]

    num_cols = 15
    num_rows = (num_channels + num_cols - 1) // num_cols
    fig, axs = plt.subplots(num_rows, num_cols, figsize=(12, 5*num_rows))

    for i in range(num_channels):
        row = i // num_cols
        col = i % num_cols

        axs[row, col].imshow(activations[0, :, :, i], cmap='hot')
        axs[row, col].axis('off')

    for i in range(num_channels, num_rows*num_cols):
        row = i // num_cols
        col = i % num_cols
        axs[row, col].axis('off')

    plt.subplots_adjust(wspace=0.1, hspace=0.1)

    plt.show()


#show maen values
fig2, axs2 = plt.subplots(1, layer_number+1)

axs2[0].imshow(test_image)
axs2[0].axis('off')
col2 = 1

for i in range(layer_number):
    activation_model = Model(inputs=network.input, outputs=network.layers[i].output)

    activations = activation_model.predict(np.expand_dims(test_image, axis=0))
    heatmap = np.mean(activations[0], axis=-1)
    axs2[col2].imshow(heatmap, cmap='hot')
    axs2[col2].axis('off')
    col2 = col2+1

plt.show()
