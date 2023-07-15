from flask import Flask, request, redirect, jsonify, render_template, send_file
from flask_cors import CORS
from werkzeug.utils import secure_filename

import cv2
import os
import torch
from facenet_pytorch import MTCNN
import matplotlib.pyplot as plt
import face_recognition

import json
import base64

app = Flask(__name__)
CORS(app)


@app.route('/recorteFacial', methods=['POST'])
def upload_file():
    try:
        # Lista para almacenar los objetos JSON
        json_list = []

        # check if the post request has the file part
        if 'file' not in request.files:
            resp = jsonify({'message' : 'No file part in the request'})
            resp.status_code = 400
            return resp
        
        if not os.path.exists("imagen"):
            os.makedirs("imagen")
            print("Nueva carpeta creada: imagen")
            
        #capturamos archivo subido
        imagesPath = request.files['file']
        #obtenemos el nombre del archivo
        filename = secure_filename(imagesPath.filename)
        #asignamos una ruta al archivo
        output_file = "./imagen/" + filename
        #guardamos el archivo subido
        imagesPath.save(output_file)

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
        #plt.show()

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
                #i += 1
                #cv2.imshow("face", cv2.cvtColor(face, cv2.COLOR_BGR2RGB))
                cv2.waitKey(0)
                
                
                #------------Codificar Code: Codificamos cada una de las imagenes------------
                output_file_faces = "./faces/" + str(i) + ".png"
                imagen_file = cv2.imread(output_file_faces)
                face_locations = face_recognition.face_locations(imagen_file)

                if len(face_locations) > 0:
                    face_loc = face_locations[0]
                
                    face_image_encodings = face_recognition.face_encodings(imagen_file, known_face_locations=[face_loc])[0]
                    print("face_image_encondings:", face_image_encodings)


                    if not os.path.exists("archivosjson"):
                        os.makedirs("archivos")

                    # Obtener el nombre base del archivo sin la extensión
                    base_name = os.path.splitext(os.path.basename(output_file_faces))[0]
                    
                    # Crear diccionario con la información de la cara
                    face_data = {
                        "encoding": face_image_encodings.tolist(),
                        'imagen': base64.b64encode(open('./faces/' + str(i) + ".png", 'rb').read()).decode('utf-8'),
                    }
                    print("face_dict:", face_data)

                    cv2.rectangle(imagen_file, (face_loc[3], face_loc[0]),(face_loc[1], face_loc[2]), (0, 255, 0))
                    
                    # Guardar diccionario en archivo json en el directorio "archivos"
                    with open(f"archivosjson/{base_name}.json", "w") as f:
                        json.dump(face_data, f)
                        
                    # Agregar el JSON a la lista
                    json_list.append(face_data)
                    i += 1
                else:
                    resp = jsonify({'server': "No se ha detectado ninguna cara en la imagen"})
                    resp.status_code = 201
                    return resp
        
        
        resp = jsonify({'server': json_list})
        resp.status_code = 201
        return resp
    
    except Exception as e:
            resp = jsonify({'error': 'Error interno del servidor', 'message': str(e)})
            resp.status_code = 500
            return resp

if __name__ == '__main__':
    app.run()