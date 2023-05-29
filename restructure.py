# source: https://medium.com/@W.rajat/training-on-large-datasets-that-dont-fit-in-memory-in-keras-60a974785d71
import os
import shutil
import numpy as np
import shutil
import cv2

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle


import keras
from keras.utils import to_categorical

def remove():
    for subdir, _, files in os.walk(fr".\ffhq-0.5T-dataset"):
        for file in files:
            if file == 'desktop.ini':
                print(f'{subdir}\{file}')
                os.remove(f'{subdir}\{file}')



def join_dirs() -> None:
    counter = 0
    for subdir, _, files in os.walk(source_path):
        for file in files:
            full_path = os.path.join(subdir, file)
            shutil.copy(full_path, destination_path)
            print(full_path)
            counter += 1
    print(f'{counter} files copied.')


def store_labels(categorical:bool = False, save:bool = True) -> tuple:
    path, subdirs, files = os.walk(destination_path).__next__()
    m = len(files)
    print(f'{m} files restructured.')
    filenames = []
    labels = np.zeros((m, 1))

    filenames_counter = 0
    for path, subdirs, files in os.walk(source_path):
        if len(files):
            if path.split('\\')[3] == 'generated':
                labels_counter = 0
            if path.split('\\')[3] == 'real':
                labels_counter = 1
            print(labels_counter)
            for file in files:
                print(file)
                filenames.append(file)
                labels[filenames_counter, 0] = labels_counter
                filenames_counter += 1
        
    print(f'{len(filenames)} filenames.')
    print(f'{labels.shape} labels. ')

    if categorical:
        labels = to_categorical(labels)
        print(labels)
    if save:
        np.save(f'{abs_path}/labels.npy', labels)
        np.save(f'{abs_path}/filenames.npy', filenames)

    return filenames, labels

def shuffle_split(filenames, labels):
    filenames_shuffled, labels_shuffled = shuffle(filenames, labels)

    # you can load them later using np.load().
    np.save(fr'{abs_path}\labels_shuffled.npy', labels_shuffled)
    np.save(fr'{abs_path}\filenames_shuffled.npy', filenames_shuffled)

    filenames_shuffled_numpy = np.array(filenames_shuffled)

    X_train_filenames, X_val_filenames, y_train, y_val = train_test_split(
        filenames_shuffled_numpy, labels_shuffled, test_size=0.2, random_state=1)

    print(fr'Train filenames shape: {X_train_filenames.shape}') 
    print(fr'Train labels shape: {y_train.shape}')
    print(fr'Test filenames shape: {X_val_filenames.shape}')
    print(fr'Test labels shape: {y_val.shape}')

    np.save(f'{abs_path}\X_train_filenames.npy', X_train_filenames)
    np.save(f'{abs_path}\y_train.npy', y_train)
    np.save(f'{abs_path}\X_val_filenames.npy', X_val_filenames)
    np.save(f'{abs_path}\y_val.npy', y_val)
    
if __name__ == '__main__':
    abs_path = fr'.\ffhq_mix'
    source_path, destination_path = fr'{abs_path}\dataset', fr'{abs_path}\restructured'
    # join_dirs()
    data = store_labels()
    shuffle_split(*data)
