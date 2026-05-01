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
train_generator = ImageDataGenerator(preprocessing_function=preprocess_input)
test_generator = ImageDataGenerator(preprocessing_function=preprocess_input)

'''Image-data-generator-flows'''
train_datagen = train_generator.flow_from_directory(directory=train_path,
                                                    target_size=(224,224),
                                                    class_mode='categorical',
                                                    batch_size=16,
                                                    shuffle=True,
                                                    )
test_datagen = test_generator.flow_from_directory(directory=test_path,
                                                    target_size=(224,224),
                                                    class_mode='categorical',
                                                    batch_size=16,
                                                    shuffle=False,
                                                    )


base_model = ResNet152(weights='imagenet',
                   input_shape=(224,224,3),
                   include_top=False,
                       )

'''defining callbacks'''
callbacks = [ReduceLROnPlateau(monitor='val_loss',
                               factor=0.45,
                               verbose=True),
             EarlyStopping(monitor='val_loss',
                           patience=6,
                           verbose=True,
                           restore_best_weights=True)]

'''freeezing layers'''
for layer in base_model.layers[:-100]:
    layer.trainable = False

'''defining functional model'''

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.5)(x)
output_layer = Dense(38,activation='softmax')(x)

'''defining model'''
model = Model(base_model.input,output_layer)

'''compiling model'''
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'],
              )

history = model.fit(x=train_datagen,
                                validation_data = test_datagen,
                               
                               epochs=20,
                               callbacks=callbacks,

                   )

model_prediction = model.predict(x=test_datagen)

confusion_matrix_output_normed = cm(test_datagen.classes,np.argmax(model_prediction,axis=1),normalize=True)
confusion_matrix_output_unnormed = cm(test_datagen.classes,np.argmax(model_prediction,axis=1),normalize=False)

print(confusion_matrix_output_normed)
print(confusion_matrix_output_unnormed)

model.save('SoftmaxClassificationModel.h5')
