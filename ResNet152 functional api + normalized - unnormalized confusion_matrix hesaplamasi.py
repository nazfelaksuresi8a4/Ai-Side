import numpy as np
from jinja2.nodes import Output
from keras.src.layers import Activation, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import  ImageDataGenerator
from tensorflow.keras.applications.resnet import  ResNet152,preprocess_input
from tensorflow.keras.layers import  (BatchNormalization,ReLU,Softmax,MaxPooling2D
                                        ,GlobalAveragePooling2D,Dropout,Input,Conv2D)
from tensorflow.keras.callbacks import  ReduceLROnPlateau,EarlyStopping
from sklearn.metrics import confusion_matrix as cm
import tensorflow as tf

train_path = r"Dataset\train"
test_path = r"Dataset\test"

'''ImageDataGenerators'''
train_generator = ImageDataGenerator(rescale=1./255,
                                     preprocessing_function=preprocess_input)
test_generator = ImageDataGenerator(rescale=1./255)

'''Image-data-generator-flows'''
train_datagen = train_generator.flow_from_directory(directory=train_path,
                                                    target_size=(224,224),
                                                    class_mode='categorical',
                                                    batch_size=16,
                                                    shuffle=True,
                                                    )
test_datagen = train_generator.flow_from_directory(directory=test_path,
                                                    target_size=(224,224),
                                                    class_mode='categorical',
                                                    batch_size=16,
                                                    shuffle=False,
                                                    )


resnet = ResNet152(weights='imagenet',
                   input_shape=(224,224,3),
                   include_top=False)

'''defining callbacks'''
callbacks = [ReduceLROnPlateau(monitor='val_loss',
                               factor=0.45,
                               verbose=True),
             EarlyStopping(monitor='val_loss',
                           patience=6,
                           verbose=True,
                           restore_best_weights=True)]

'''freeezing layers'''
for layer in resnet.layers[:-100]:
    layer.trainable = False

'''defining functional model'''
input_layer = Input(shape=(224,224,3))

x = Conv2D(32,(3,3),strides=(1,1))(input_layer)
x = Activation(activation='relu')(x)
x = BatchNormalization()(x)
x = MaxPooling2D((2,2),strides=(1,1))(x)

x = Conv2D(16,(3,3),strides=(1,1))(x)
x = Activation(activation='relu')(x)
x = BatchNormalization()(x)
x = MaxPooling2D((2,2),strides=(1,1))(x)

x = Conv2D(8,(3,3),strides=(1,1))(x)
x = Activation(activation='relu')(x)
x = BatchNormalization()(x)
x = MaxPooling2D((2,2),strides=(1,1))(x)

x = Conv2D(4,(3,3),strides=(1,1))(x)
x = Activation(activation='relu')(x)
x = BatchNormalization()(x)
x = MaxPooling2D((2,2),strides=(1,1))(x)

x = Dropout(0.6)(x)
x = GlobalAveragePooling2D()(x)
output_layer = Dense(38,activation='softmax')(x)

'''defining model'''
model = Model(input_layer,output_layer)

'''compiling model'''
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'],
              )

model_prediction = model.fit(x=train_datagen,
                   batch_size=32,
                   epochs=20,
                   callbacks=callbacks,

                   )

confusion_matrix_output_normed = cm(model_prediction,test_datagen,normalize=True)
confusion_matrix_output_unnormed = cm(model_prediction,test_datagen,normalize=False)

print(confusion_matrix_output_normed)
print(confusion_matrix_output_unnormed)

model.save('SoftmaxClassificationModel.h5')
