#---

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

#---

import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
import numpy as np
import opendatasets as od
import pandas as pd
import keras
from keras.optimizers import Adam
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.inception_resnet_v2 import preprocess_input, decode_predictions
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.applications import InceptionResNetV2

#---

od.download("https://www.kaggle.com/datasets/gonzajl/tumores-cerebrales-mri-dataset")

#---

datos = pd.read_csv('datos.csv')

#---

cant_max = 3250

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

imagenes, etiquetas = cargar_imagenes(paths, tags)

#---

div_test = int(len(imagenes) * 0.8)

#---

imagenes_entrenamiento, imagenes_prueba = imagenes[:div_test], imagenes[div_test:]
etiquetas_entrenamiento, etiquetas_prueba = etiquetas[:div_test], etiquetas[div_test:]

#---

# Convertir las listas a arreglos NumPy si no están en ese formato
imagenes_entrenamiento = np.array(imagenes_entrenamiento)
etiquetas_entrenamiento = np.array(etiquetas_entrenamiento)

# Asegurar que las etiquetas estén en el formato correcto si son una lista de listas
etiquetas_entrenamiento = np.array([np.array(etiqueta) for etiqueta in etiquetas_entrenamiento])

#---

imagenes_entrenamiento = np.array(imagenes_entrenamiento).astype(float) / 255
imagenes_prueba = np.array(imagenes_prueba).astype(float) / 255

etiquetas_entrenamiento = np.array(etiquetas_entrenamiento)
etiquetas_prueba = np.array(etiquetas_prueba)

#---

# Cargar el modelo InceptionResNetV2 preentrenado
base_model = InceptionResNetV2(include_top=False, weights='imagenet', input_shape=(224, 224, 3))

# Agregar una capa de agrupación global promedio
x = GlobalAveragePooling2D()(base_model.output)

# Agregar la nueva capa densa para la clasificación con 4 clases y activación softmax
predictions = Dense(4, activation='softmax')(x)

# Crear un nuevo modelo combinando el modelo base y las nuevas capas
InceptionResNet = Model(inputs=base_model.input, outputs=predictions)

#---

InceptionResNet.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['categorical_accuracy'])

#---

historial = InceptionResNet.fit(imagenes_entrenamiento, etiquetas_entrenamiento, epochs=1, validation_data=(imagenes_prueba, etiquetas_prueba))

#---

def es_correcta(prediccion, esperado):
    return prediccion.index(max(prediccion)) == esperado.index(max(esperado))

#---

correctas_segun_tipo = [0, 0, 0, 0]
falladas_segun_tipo = [0, 0, 0, 0]

predicciones = InceptionResNet.predict(imagenes_prueba)
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