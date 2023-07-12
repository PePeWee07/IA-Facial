from flask import Flask, request, redirect, jsonify, render_template, send_file
from flask_cors import CORS
from werkzeug.utils import secure_filename

import cv2
import os
import torch
from facenet_pytorch import MTCNN
import matplotlib.pyplot as plt
import face_recognition

app = Flask(__name__)
CORS(app)


@app.route('/recorteFacial', methods=['POST'])
def upload_file():

    # check if the post request has the file part
    if 'file' not in request.files:
        resp = jsonify({'message' : 'No file part in the request'})
        resp.status_code = 400
        return resp
    
    imagesPath = request.files['file']
    filename = secure_filename(imagesPath.filename)
    print('Name Image: ' + filename)
    output_file = "./imagen/" + filename
    print('Ruta: ', "./imagen/" + filename)
    imagesPath.save(output_file)

    # Comprobar si se seleccionó un archivo
    if imagesPath.filename == '':
        resp = jsonify({'message' : 'No file selected for uploading'})
        resp.status_code = 400
        return resp
    

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
    image = cv2.imread(output_file)

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

    resp = jsonify({'message': 'Archivo recibido y procesado correctamente'})
    resp.status_code = 201
    return resp

if __name__ == '__main__':
    app.run()