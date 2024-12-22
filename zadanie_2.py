from operator import indexOf

import numpy as np
import os
import PIL
from PIL import Image

import tensorflow as tf
import tensorflow_datasets as tfds

model_convolutional = tf.keras.models.load_model('saved_models\convolutional_model.keras')

model_convolutional.summary()

while True:
    print("Choose file to verify your model on it. Files available:")
    samples_list = os.listdir('manual_testing_samples')
    print(samples_list)
    choose = input("Enter number of a sample you want to verify with the model:")
    print("You have choosen number " + str(choose) + " which is " + samples_list[int(choose)])

    img = Image.open("manual_testing_samples\\" + str(samples_list[int(choose)])).convert('L').resize((28, 28))
    img = np.array(img)
    predictions = model_convolutional.predict(img[None,:,:])
    print(predictions)
    print(np.max(predictions))
    most_probable=np.where(predictions == np.max(predictions))
    print("The most probable prediction is: " + str(most_probable[1]))
    print('''
*******************************************************
NEXT PREDICTION
*******************************************************''')



