import cv2
import os
import torch
from facenet_pytorch import MTCNN
import matplotlib.pyplot as plt
import face_recognition

imagesPath = "C:/Users/gusta/Desktop/ARTICULO CIENTIFICO/R_F2.0/imagen/tavo1.png"

if not os.path.exists("faces"):
    os.makedirs("faces")
    print("Nueva carpeta creada: faces")

# Detectar facial
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Running on device: {}'.format(device))

# Detector MTCNN
mtcnn = MTCNN(
    select_largest=True,
    min_face_size=20,
    thresholds=[0.6, 0.7, 0.7],
    post_process=False,
    image_size=160,
    device=device
)

# Cargar imagen
image = cv2.imread(imagesPath)

# Convertir imagen de BGR a RGB
print(image.shape)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
plt.imshow(image)
plt.show()

# Detectar caras en la imagen
boxes, _ = mtcnn.detect(image)


# Aquí se realiza el corte de las imágenes
if boxes is not None:
    i = 1
    for box in boxes:
        box = [int(coord) for coord in box]
        face_image_encodings = face_recognition.face_encodings(image, known_face_locations=[box])[0]
        print(face_image_encodings)
        face = image[box[1]:box[3], box[0]:box[2], :]
        cv2.imwrite("faces/" + str(i) + ".png", cv2.cvtColor(face, cv2.COLOR_BGR2RGB))
        i += 1
        cv2.imshow("face", cv2.cvtColor(face, cv2.COLOR_BGR2RGB))
        cv2.waitKey(0)
