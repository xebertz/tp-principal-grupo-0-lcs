#---

import os
import cv2
import numpy as np
import tensorflow as tf
import opendatasets as od
import pandas as pd
from PIL import Image
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
#---

od.download("https://www.kaggle.com/datasets/gonzajl/tumores-cerebrales-mri-dataset")
datos = pd.read_csv("./datos.csv")

#---

cant_max = 4000

glioma_df = datos[datos['glioma'] == 1].head(cant_max)
meningioma_df = datos[datos['meningioma'] == 1].head(cant_max)
pituitary_df = datos[datos['pituitary'] == 1].head(cant_max)
no_tumor_df = datos[datos['no_tumor'] == 1].head(cant_max)

datos = pd.concat([glioma_df, meningioma_df, pituitary_df, no_tumor_df], ignore_index=True)

#---

datos = datos.sample(frac=1).reset_index(drop=True)

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

#---

paths = datos.iloc[:, 0]
tags = datos.iloc[:, 1:]   #ignoro filas particulares y traigo las columnas de 1 a fin

#---

magenes, etiquetas = cargar_imagenes(paths, tags)

#---

div_test = int(len(imagenes) * 0.8)

#---

imagenes_entrenamiento, imagenes_prueba = imagenes[:div_test], imagenes[div_test:]
etiquetas_entrenamiento, etiquetas_prueba = etiquetas[:div_test], etiquetas[div_test:]

#---

print(len(imagenes_entrenamiento)) #esto no es necesario pero muestra la cantidad de elementos en cada array.
print(len(imagenes_prueba))

print(len(etiquetas_entrenamiento))
print(len(etiquetas_prueba))

#---

imagenes_entrenamiento = np.array(imagenes_entrenamiento).astype(float) / 255
imagenes_prueba = np.array(imagenes_prueba).astype(float) / 255

etiquetas_entrenamiento = np.array(etiquetas_entrenamiento)
etiquetas_prueba = np.array(etiquetas_prueba)

#---

forma_etiqueta = etiquetas_entrenamiento[0].shape #Esto muestra el formato actual de las etiquetas, en particular muestra la primera
                                                    #del array de entrenamiento
print(forma_etiqueta)

#---

modelo_cnn = tf.keras.models.Sequential([ 
    tf.keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=(224, 224, 3)),
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
                   metrics=['categorical_accuracy'])

#---

print("Entrenando modelo convolucional...")
historial_cnn = modelo_cnn.fit(imagenes_entrenamiento, etiquetas_entrenamiento, epochs=5, validation_data=(imagenes_prueba, etiquetas_prueba), use_multiprocessing=True, shuffle=True)
print("Modelo convolucional entrenado!")

#---

def es_correcta(prediccion, esperado):
    return prediccion.index(max(prediccion)) == esperado.index(max(esperado))

#---

correctas_segun_tipo = [0, 0, 0, 0]
falladas_segun_tipo = [0, 0, 0, 0]

predicciones = modelo_cnn.predict(imagenes_prueba_resized)
print(f"Cantidad de predicciones: {len(predicciones)}")

for i in range(len(predicciones)):
    prediccion = list(predicciones[i])
    index = prediccion.index(max(prediccion))
    
    if es_correcta(prediccion, list(etiquetas_prueba[i])):
        correctas_segun_tipo[index] += 1  
    else: 
        falladas_segun_tipo[index] += 1
        
cant_totales = list(map(lambda x, y: x + y, correctas_segun_tipo, falladas_segun_tipo))
print("Etiquetas:   [G,  M,  P,  N]")
print(f"Total:       {cant_totales}")
print(f"Correctas:   {correctas_segun_tipo}")
print(f"Incorrectas: {falladas_segun_tipo}")

#---