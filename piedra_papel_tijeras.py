import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import cv2
import numpy as np
from keras.layers.regularization.spatial_dropout3d import Dropout


datos, metadatos = tfds.load("rock_paper_scissors", as_supervised = True, with_info = True)

metadatos

datos_entrenamiento = []

TAMANO_IMAGEN = 100

for i, (imagen, etiqueta) in enumerate(datos["train"]):
  imagen = cv2.resize(imagen.numpy(), (TAMANO_IMAGEN, TAMANO_IMAGEN)) # cambiamos de tamaño la imagen i
  imagen = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY) # la pasamos a blanco y negro
  imagen = imagen.reshape(TAMANO_IMAGEN, TAMANO_IMAGEN, 1)  # un canal, es decir, es escala de grises
  datos_entrenamiento.append([imagen, etiqueta])
  
X = [] # Ejemplos
y = [] # Las respectivas etiquetas

for imagen, etiqueta in datos_entrenamiento:
  X.append(imagen)
  y.append(etiqueta)

# Normalización

X = np.array(X).astype(float)/255

y = np.array(y)

y

X.shape




modeloCNN = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation="relu", input_shape=(100, 100, 1)),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(64, (3,3), activation="relu"),
    tf.keras.layers.MaxPooling2D(2,2),    
    tf.keras.layers.Conv2D(128, (3,3), activation="relu"),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(100, activation = "relu"),
    tf.keras.layers.Dense(1, activation = "sigmoid")
])


modeloCNN.compile(
    optimizer = "adam",
    loss = "binary_crossentropy",
    metrics = ["accuracy"]
)


modeloCNN.fit(
    X, y,
    validation_split = 0.13,
    epochs = 20
)



modeloCNN.save("piedra_papel_tijeras_CNN.h5")