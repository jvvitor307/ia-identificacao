# Preparar os dados de entrada
img_path = 'image copy 4.png'  # Substitua pelos seus dados de entrada
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np

# Carregar o modelo
model = tf.keras.models.load_model('modelo/model.h5')
# Caminho da imagem


# Carregar a imagem com o tamanho esperado pelo modelo
img = image.load_img(img_path, target_size=(128, 128))  # Altere o tamanho conforme necessário
img_array = image.img_to_array(img)

# Expandir as dimensões para corresponder ao formato esperado pelo modelo
img_array = np.expand_dims(img_array, axis=0)

# Normalizar a imagem se necessário
img_array /= 255.0  # Normalização comum, ajuste conforme necessário

# Fazer a previsão
predictions = model.predict(img_array)

# Ver o resultado
print(predictions)

