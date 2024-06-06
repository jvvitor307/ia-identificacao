import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
model = tf.keras.models.load_model('App/modelo/model.h5')

def predict(img_path):
    
    
    img = image.load_img(img_path, target_size=(500, 500))  
    
    img_array = image.img_to_array(img)

    img_array = np.expand_dims(img_array, axis=0)

    img_array /= 255.0  

    predictions = model.predict(img_array)
    
    return predictions