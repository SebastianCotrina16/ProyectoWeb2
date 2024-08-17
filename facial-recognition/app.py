"""
Este archivo define una aplicación Flask
que proporciona una API para el registro
y reconocimiento facial de usuarios.
Utiliza modelos de Deep Learning para detectar
y comparar descriptores faciales
y almacena la información de los usuarios en una base de datos.
"""

import os
from flask_talisman import Talisman
from flask_cors import CORS
from flask import Flask, request, jsonify
import cv2
import numpy as np
import dlib
from models import User,  SessionLocal #pylint: disable=E0401

app = Flask(__name__)

CORS(app, resources={r"/*": {"origins": "http://localhost:3000"}})

talisman = Talisman(app, content_security_policy=None, force_https=False)

IMAGE_FOLDER = 'faces'
if not os.path.exists(IMAGE_FOLDER):
    os.makedirs(IMAGE_FOLDER)

detector = dlib.get_frontal_face_detector() #pylint: disable=E1101
sp = dlib.shape_predictor('models/shape_predictor_68_face_landmarks.dat') #pylint: disable=E1101
facerec = dlib.face_recognition_model_v1('models/dlib_face_recognition_resnet_model_v1.dat') #pylint: disable=E1101

FACE_MATCH_TOLERANCE = 0.6

def extract_face_descriptor(image):
    """Detecta y extrae un descriptor de rostro de una imagen."""
    dets = detector(image, 1)
    if len(dets) > 0:
        shape = sp(image, dets[0])
        face_descriptor = facerec.compute_face_descriptor(image, shape)
        return np.array(face_descriptor)
    return None

def is_face_match(face_descriptor, known_face_descriptor, tolerance=FACE_MATCH_TOLERANCE):
    """Compara dos descriptores faciales y determina si son coincidentes según una tolerancia."""
    return np.linalg.norm(face_descriptor - known_face_descriptor) <= tolerance

def find_user_by_face_descriptor(face_descriptor):
    """Busca un usuario en la base de datos utilizando un descriptor facial."""
    with SessionLocal() as session:
        users = session.query(User).all()
        for user in users:
            known_face_descriptor = np.fromstring(user.face_descriptor, sep=',')
            if is_face_match(face_descriptor, known_face_descriptor):
                return user
    return None

def add_user_to_db(dni, name, face_descriptor, image_path):
    """Añade un nuevo usuario a la base de datos."""
    face_descriptor_str = ','.join(map(str, face_descriptor))
    with SessionLocal() as session:
        new_user = User(
            dni=dni,
            name=name,
            face_descriptor=face_descriptor_str,
            image_path=image_path
        )
        session.add(new_user)
        session.commit()

@app.route('/recognize', methods=['POST'])
def recognize():
    """Procesa la imagen enviada y reconoce al usuario en la base de datos."""
    image = request.files['image'].read()
    image_array = np.frombuffer(image, np.uint8)
    img_np = cv2.imdecode(image_array, cv2.IMREAD_COLOR) #pylint: disable=E1101

    face_descriptor = extract_face_descriptor(img_np)
    if face_descriptor is not None:
        user = find_user_by_face_descriptor(face_descriptor)
        if user:
            print("User found")
            return jsonify({'user': {'name': user.name, 'dni': user.dni}})
    return jsonify({'user': None})

@app.route('/register', methods=['POST'])
def register():
    """Registra un nuevo usuario utilizando su imagen facial."""
    dni = request.form['dni']
    name = request.form['name']
    image = request.files['image'].read()
    image_array = np.frombuffer(image, np.uint8)
    img_np = cv2.imdecode(image_array, cv2.IMREAD_COLOR) #pylint: disable=E1101
    print(img_np.shape)
    print("Working")
    face_descriptor = extract_face_descriptor(img_np)
    if face_descriptor is not None:
        image_path = os.path.join(IMAGE_FOLDER, f"{dni}.jpg")
        cv2.imwrite(image_path, img_np) #pylint: disable=E1101

        add_user_to_db(dni, name, face_descriptor, image_path)
        print("User added")
        return jsonify({'success': True})
    print("User not added")
    return jsonify({'success': False})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)
