import cv2

def abrirCamera():
# Inicializa a câmera
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Erro ao abrir a câmera")
        exit()
    return cap
        
def lerCamera(cap):
    # Lê um frame da câmera
    ret, frame = cap.read()
    return frame, ret
    
def salvarImage(frame,ret,caminho):
    
    if ret:
        # Salva a imagem no disco
        cv2.imwrite(caminho, frame)
    else:
        return "Erro ao capturar a imagem"
    
def liberaCamera(cap):
# Libera a câmera e fecha a janela
    cap.release()
    cv2.destroyAllWindows()
