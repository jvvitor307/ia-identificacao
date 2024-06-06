from camera import *
from predict import *
import time
import os



cap = abrirCamera()
verdade = True
cont = 0
while verdade:
   
    cont+=1
    frame, ret = lerCamera(cap)
    erro = salvarImage(frame,ret,f'treinamento/pessoas/pessoa/imagemCapturada.jpeg')
    predic = predict(f'treinamento/pessoas/pessoa/imagemCapturada.jpeg')
    print(predic)
    if (predic < 0.7):
        os.system('xdg-screensaver lock') 
        
    

