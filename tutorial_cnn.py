# Tutorial Source: 
# https://www.deeplearningbook.com.br/reconhecimento-de-imagens-com-redes-neurais-convolucionais-em-python-parte-4/


import tensorflow as tf
import keras as ka
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
from keras.preprocessing import image

# Inicializando a Rede Neural Convolucional
classifier = Sequential()

# Passo 1 - Primeira Camada de Convolução
classifier.add(Conv2D(32, (3, 3), input_shape = (64, 64, 3), activation = 'relu'))

# Passo 2 - Pooling
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Adicionando a Segunda Camada de Convolução
# classifier.add(Conv2D(32, (3, 3), activation = 'relu'))

# classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Passo 3 - Flattening
classifier.add(Flatten())

# Passo 4 - Full connection
classifier.add(Dense(units = 128, activation = 'relu'))
classifier.add(Dense(units = 1, activation = 'sigmoid'))

# Compilando a rede
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Criando os objetos train_datagen e validation_datagen com as regras de pré-processamento das imagens
train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

validation_datagen = ImageDataGenerator(rescale = 1./255)


# Pré-processamento das imagens de treino e validação
# train_path = 'dogs_cats.nosync/dataset_train'
train_path = 'homer_bart.nosync/training_set'
training_set = train_datagen.flow_from_directory(train_path,
                                                 target_size = (64, 64),
                                                 batch_size = 32,
                                                 class_mode = 'binary')

validation_set = validation_datagen.flow_from_directory('homer_bart.nosync/test_set',
                                                        target_size = (64, 64),
                                                        batch_size = 32,
                                                        class_mode = 'binary')


# Executando o treinamento (esse processo pode levar bastante tempo, dependendo do seu computador)
# classifier.fit_generator(training_set,
#                          steps_per_epoch = 500,
#                          epochs = 5)


classifier.fit_generator(training_set,
                         epochs = 10,
                         validation_data = validation_set)


# # Testes
# labels = ["cat", "dog"]
# hits = 0
# errors = 0 

# test_path = "dogs_cats.nosync/dataset_test/"

# for label in labels:
#     for n in range(500):
#         test_image = image.load_img(f"{test_path}{label}.{n}.jpg", target_size = (64, 64))
#         test_image = image.img_to_array(test_image)
#         test_image = np.expand_dims(test_image, axis = 0)
#         result = classifier.predict(test_image)
#         training_set.class_indices

#         if result[0][0] == 1:
#             prediction = 'dog'
#         else:
#             prediction = 'cat'

#         if(prediction == label):
#             hits += 1
#         else:
#             errors += 1

# print(f"From: {hits+errors}, {errors} errors and {hits} hits")
