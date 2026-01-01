import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.models import Sequential
import matplotlib.pyplot as plt

# ===================== DATA PATH =====================
root_path = r"C:\Users\alper\Desktop\PlantVillage"
train_path = root_path + r"\train"
test_path = root_path + r"\test"

datagen = ImageDataGenerator(rescale=1./255)

train_data = datagen.flow_from_directory(directory=train_path,
                                         target_size=(224,224,3),
                                         batch_size=16,
                                         )

test_data = datagen.flow_from_directory(directory=test_path,
                                        target_size=(224,224,3),
                                        batch_size=16)

