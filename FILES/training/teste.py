import numpy as np
import tensorflow as tf
#from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

model = tf.keras.models.load_model("FILES/training/trained_model.keras")

model.summary()