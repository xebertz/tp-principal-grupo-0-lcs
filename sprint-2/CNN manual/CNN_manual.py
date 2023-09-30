import os
import cv2
import numpy as np
import tensorflow as tf
import opendatasets as od
import pandas as pd
from PIL import Image
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical

#---

od.download("https://www.kaggle.com/datasets/gonzajl/tumores-cerebrales-mri-dataset")
datos = pd.read_csv("./datos.csv")

#---

def cargar_imagenes(nombres, et):
    imagenes = []
    etiquetas = []
    for i in range(len(nombres)):
        ruta = 'tumores-cerebrales-mri-dataset/' + nombres[i][2:]
        imagen = cv2.imread(ruta, cv2.IMREAD_COLOR)
        imagenes.append(imagen)
        etiquetas.append(list(et.iloc[i]))
        
    return imagenes, etiquetas


paths = datos.iloc[:, 0]
tags = datos.iloc[:, 1:]   #ignoro filas particulares y traigo las columnas de 1 a fin

#---

magenes, etiquetas = cargar_imagenes(paths, tags)

#---

imagenes_entrenamiento, imagenes_prueba = imagenes[:3000], imagenes[:1000]  # el maximo aca es 33000 de entrenamiento y 9000 de prueba
etiquetas_entrenamiento, etiquetas_prueba = etiquetas[:3000], etiquetas[:1000] # el maximo aca es 33000 de entrenamiento y 9000 de prueba

#---

imagenes_entrenamiento_resized = np.array([cv2.resize(img, (100, 100)) for img in imagenes_entrenamiento])
imagenes_prueba_resized = np.array([cv2.resize(img, (100, 100)) for img in imagenes_prueba])

#---

imagenes_entrenamiento = np.array(imagenes_entrenamiento).astype(float) / 255
imagenes_prueba = np.array(imagenes_prueba).astype(float) / 255

etiquetas_entrenamiento = np.array(etiquetas_entrenamiento)
etiquetas_prueba = np.array(etiquetas_prueba)

#---

# Asumiendo que tus etiquetas son listas de listas con valores 0 o 1
etiquetas_entrenamiento = to_categorical([clase[0] for clase in etiquetas_entrenamiento], num_classes=4)
etiquetas_prueba = to_categorical([clase[0] for clase in etiquetas_prueba], num_classes=4)

# Imprime las formas para verificar
print("Forma de etiquetas_entrenamiento después de to_categorical:", etiquetas_entrenamiento.shape)
print("Forma de etiquetas_prueba después de to_categorical:", etiquetas_prueba.shape)

#---

forma_etiqueta = etiquetas_entrenamiento[0].shape
print(forma_etiqueta)

#---

modelo_cnn = tf.keras.models.Sequential([ 
    tf.keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=(100, 100, 3)),
    tf.keras.layers.MaxPooling2D(3, 3),
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(4, activation='softmax')  # 4 clases, activación softmax
])

#---

modelo_cnn.compile(optimizer='adam',
                   loss='categorical_crossentropy',  # función de pérdida para clasificación multiclase
                   metrics=['accuracy'])

#---

print("Entrenando modelo convolucional...")
historial_cnn = modelo_cnn.fit(imagenes_entrenamiento_resized, etiquetas_entrenamiento, epochs=5, validation_data=(imagenes_prueba_resized, etiquetas_prueba), use_multiprocessing=True, shuffle=True)
print("Modelo convolucional entrenado!")