import cv2
import face_recognition
import json
import os

image = "faces/chaly.png"
imagen_file = cv2.imread(image)
face_locations = face_recognition.face_locations(imagen_file)

if len(face_locations) > 0:
    face_loc = face_locations[0]
   
    face_image_encodings = face_recognition.face_encodings(imagen_file, known_face_locations=[face_loc])[0]
    print("face_image_encondings:", face_image_encodings)

    # Crear diccionario con la información de la cara
    face_data = {
        "encodings": face_image_encodings.tolist()  
    }
    print("face_dict:", face_data)

    if not os.path.exists("archivosjson"):
        os.makedirs("archivos")

    # Obtener el nombre base del archivo sin la extensión
    base_name = os.path.splitext(os.path.basename(image))[0]

    # Guardar diccionario en archivo json en el directorio "archivos"
    with open(f"archivosjson/{base_name}.json", "w") as f:
        json.dump(face_data, f)

    cv2.rectangle(imagen_file, (face_loc[3], face_loc[0]),(face_loc[1], face_loc[2]), (0, 255, 0))
else:
    print("No se ha detectado ninguna cara en la imagen")



