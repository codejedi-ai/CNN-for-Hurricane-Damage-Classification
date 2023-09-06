
import urllib
import zipfile

import tensorflow as tf

def download_and_extract_data():
    url = 'https://storage.googleapis.com/download.tensorflow.org/data/certificate/satellitehurricaneimages.zip'
    urllib.request.urlretrieve(url, 'satellitehurricaneimages.zip')
    with zipfile.ZipFile('satellitehurricaneimages.zip', 'r') as zip_ref:
        zip_ref.extractall()

def preprocess(image, label):
    image = image / .255
    return image, label

def solution_model():
    download_and_extract_data()

    IMG_SIZE = 128
    BATCH_SIZE = 64
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        directory='train/',
        image_size=(IMG_SIZE,IMG_SIZE),
        batch_size=BATCH_SIZE)

    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        directory='validation/',
        image_size=(IMG_SIZE,IMG_SIZE),
        batch_size=BATCH_SIZE)

    train_ds = train_ds.map(
        preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE).prefetch(
        tf.data.experimental.AUTOTUNE)
    val_ds = val_ds.map(
        preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    # Code to define the model
    model = None
    history = None

    return model, history

