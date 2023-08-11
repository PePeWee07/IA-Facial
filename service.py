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

import numpy as np
import pymongo

app = Flask(__name__)
CORS(app, resources={
    r"/guardar": {"origins": "http://localhost:4200"},
    r"/reconocer": {"origins": "http://localhost:4200"},
    r"/recorteFacial": {"origins": "http://localhost:4200"},
    r"/borrar_contenido": {"origins": "http://localhost:4200"}
})

# Conectarse a la base de datos de MongoDB te quiero mucho <3
client = pymongo.MongoClient('mongodb://localhost:27017')
database = client['reconocimiento']
collection = database['personas']

# Definir la función de similitud de cosenos
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# Vectorizar la función de similitud de cosenos
vectorized_cosine_similarity = np.vectorize(cosine_similarity, signature='(n),(m)->()')

#Medoto par aguardar en mongodb
def guardar_objeto_en_mongodb(data):
    try:
        # guardamos en la base de datos y retornamos el ID
        object_id = collection.insert_one(data).inserted_id
        return str(object_id)
    except Exception as e:
        # error al guardar el objeto en MongoDB
        print(f"Error al guardar el objeto en MongoDB: {e}")
        return None

@app.route('/guardar', methods=['POST'])
def create_object():
    data = request.get_json()
    object_id = guardar_objeto_en_mongodb(data)

    if object_id:
        return jsonify({"message": "Objeto guardado correctamente", "object_id": object_id}), 201
    else:
        return jsonify({"message": "Error al guardar el objeto"}), 500
    
def borrar_contenido_carpeta(carpeta):
    # Verificar si la carpeta existe
    if os.path.exists(carpeta):
        # Obtener lista de archivos en la carpeta
        archivos = os.listdir(carpeta)

        # Borrar cada archivo en la carpeta
        for archivo in archivos:
            ruta_archivo = os.path.join(carpeta, archivo)
            os.remove(ruta_archivo)
    else:
        print(f"La carpeta {carpeta} no existe")

@app.route('/borrar_contenido', methods=['GET'])
def borrar_contenido():
    try:
        # Rutas de las dos carpetas que quieres vaciar
        carpeta1 = "./faces"
        carpeta2 = "./archivosjson"
        carpeta3 = "./imagen"

        # Borrar contenido de las dos carpetas
        borrar_contenido_carpeta(carpeta1)
        borrar_contenido_carpeta(carpeta2)
        borrar_contenido_carpeta(carpeta3)

        # Devolver respuesta de éxito
        resp = jsonify({'server': "Clean Server"})
        resp.status_code = 201
        return resp
    
    except Exception as e:
        # Devolver respuesta en caso de error
        resp = jsonify({'server': "Error al limpiar el servidor", 'message': str(e)})
        resp.status_code = 201
        return resp


@app.route('/reconocer', methods=['POST', 'OPTIONS'])
def reconocer():
    try:
        if request.method == 'OPTIONS':
            # Handle preflight request
            response = app.make_default_options_response()
        else:
            input_encoding_str = request.json['inputEncodingStr']

            # Obtener el encoding de entrada
            input_encoding = np.array(json.loads(input_encoding_str))

            # Obtener todos los encodings de la colección
            encodings = []
            for doc in collection.find({}, {'encoding': 1}):
                encoding_str = doc['encoding']
                encoding = np.array(json.loads(encoding_str))
                encodings.append(encoding)

            # Calcular las similitudes entre el encoding de entrada y todos los encodings de la colección
            similarities = vectorized_cosine_similarity(input_encoding, np.array(encodings))

            # Encontrar la mejor similitud (similitud máxima)
            best_similarity_index = np.argmax(similarities)
            best_similarity = similarities[best_similarity_index]

            # Verificar si la mejor similitud supera el umbral
            if best_similarity > 0.92:
                doc = collection.find_one({'encoding': json.dumps(encodings[best_similarity_index].tolist())})
                result = {
                    'similitud': best_similarity,
                    'nombre': doc['nombre'],
                    'apellido': doc['apellido']
                }
            else:
                result = None

            response = jsonify({'server': result})

        # Agregar encabezados CORS a la respuesta
        response.headers['Access-Control-Allow-Origin'] = '*'
        response.headers['Access-Control-Allow-Methods'] = 'POST, OPTIONS'
        response.headers['Access-Control-Allow-Headers'] = 'Content-Type'

        resp = response
        resp.status_code = 201
        return resp
    
    except Exception as e:
        response = jsonify({'error': "Error al comparar", 'message': str(e)})
        response.status_code = 500
        return response


@app.route('/recorteFacial', methods=['POST'])
def upload_file():
    try:

        # Lista para almacenar los objetos JSON---
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
        ruta_imagen  = "./imagen/" + filename
        #guardamos el archivo subido
        imagesPath.save(ruta_imagen )

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
        image = cv2.imread(ruta_imagen)

        if image is None:
            print(f"Error al leer la imagen en la ruta: {ruta_imagen}")
            resp = jsonify({'error': 'Error al leer la imagen en la ruta'})
            resp.status_code = 500
            return resp

        # Convertir imagen de BGR a RGB
        print(image.shape)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Detectar caras en la imagen
        boxes, _ = mtcnn.detect(image)

        # Aquí se realiza el corte de las imágenes
        if boxes is not None:
            i = 1
            for box in boxes:
                box = [int(coord) for coord in box]
                face = image[box[1]:box[3], box[0]:box[2], :]
                cv2.imwrite("faces/" + str(i) + ".png", cv2.cvtColor(face, cv2.COLOR_BGR2RGB))
                i += 1
            
            a = 1
            for contador in boxes:
                output_file_faces = "./faces/" + str(a) + ".png"
                print("output_file_faces: ", output_file_faces)
                imagen_file = cv2.imread(output_file_faces)

                face_locations = face_recognition.face_locations(imagen_file,  number_of_times_to_upsample=2, model="hog")
                print("FaceLocation: ", len(face_locations))


                imgBase64 = base64.b64encode(open(output_file_faces, 'rb').read()).decode('utf-8')
                if len(face_locations) > 0:
                    face_loc = face_locations[0]
                    face_image_encodings = face_recognition.face_encodings(imagen_file, known_face_locations=[face_loc])[0]
                

                    # Crear diccionario con la información de la cara
                    face_data = {
                        "encodings": face_image_encodings.tolist(),
                        'imagen': imgBase64,  
                    }
                    #print("face_dict:", face_data)

                    if not os.path.exists("archivosjson"):
                        os.makedirs("archivos")

                    # Obtener el nombre base del archivo sin la extensión
                    # base_name = os.path.splitext(os.path.basename(image))[0]
                    base_name = str(a)

                    # Guardar diccionario en archivo json en el directorio "archivos"
                    with open(f"archivosjson/{base_name}.json", "w") as f:
                        json.dump(face_data, f)

                    # Agregar el JSON a la lista
                        json_list.append(face_data)

                else:
                    print("No se ha detectado ninguna cara en la imagen")
                    # imgBase64 = base64.b64encode(open(output_file_faces, 'rb').read()).decode('utf-8')
                    face_data = {
                        "encodings": [0],
                        'imagen': imgBase64,  
                    }
                    json_list.append(face_data)
                a += 1

        resp = jsonify({'server': json_list})
        resp.status_code = 201
        return resp
    
    except Exception as e:
            resp = jsonify({'error': 'Error interno del servidor', 'message': str(e)})
            resp.status_code = 500
            return resp

if __name__ == '__main__':
    app.run()