from deepface import DeepFace
from PIL import Image
import cv2

# Utiliza o construtor CascadeClassifier para criar o objeto df, que guarda as funções de um algoritmo de classificação em cascata
df = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Cria o objeto de camera considerando um arquivo de vídeo
# camera = cv2.VideoCapture("video.mp4")

# Cria o objeto camera considerando a webcam do computador
camera = cv2.VideoCapture(0)
image = ""

while True:
    # A função .read() retorna dois valores: 1- Um valor booleano identificando se um frame foi encontrado; 
    # 2- Uma matriz de pixels representando o frame do vídeo que foi encontrado
    (sucesso, frame) = camera.read()
   
    if not sucesso: #final do video
        print(sucesso)
        break
    
    # Converte o frame para tons de cinza
    frame_pb = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    
    # Detecta as faces no frame
    faces = df.detectMultiScale(frame_pb, scaleFactor = 1.1, minNeighbors=15, minSize=(30,30), flags=cv2.CASCADE_SCALE_IMAGE) 
    
    # Faz uma cópia do frame
    frame_temp = frame.copy()
    image = frame_temp

    # Desenha retângulos amarelos, na imagem frame_temp, nas posições onde foram encontradas faces
    for (x, y, lar, alt) in faces:
        cv2.rectangle(frame_temp, (x, y), (x + lar, y + alt), (0, 255, 255), 2) 
         
    cv2.imshow("Encontrando faces...", frame_temp) 
    
    # Espera que a tecla 's' seja pressionada para sair 
    if cv2.waitKey(1) & 0xFF == ord("s"): 
        break


# Desaloca a memória do objeto camera e fecha todas as janelas abertas pela biblioteca OpenCV 
camera.release()
cv2.destroyAllWindows()
# mostra o último frame que o reconhecedor de rostos salvou
cv2.imshow("ultimo frame", image)
cv2.waitKey(0)

#verifica se as duas imagens são iguais
result = DeepFace.verify(
  img1_path = image,
  img2_path = "teste.jpg",
)
print(result['verified'])
