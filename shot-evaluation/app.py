"""
A Flask application for detecting objects in images using a pre-trained Faster R-CNN model.
Handles file uploads, performs object detection, and records results in a SQL database.
"""

import os
from datetime import datetime
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
from PIL import Image
import torch
import torchvision.transforms as T
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import pyodbc

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

CONNECTION_STRING= (
    'DRIVER={ODBC Driver 17 for SQL Server};'
    'SERVER=svrweb2frevb.database.windows.net;'
    'DATABASE=sistemabalas;'
    'UID=farviveros;'
    'PWD=Emmanuel$123;'
    'Encrypt=yes;'
    'TrustServerCertificate=no;'
    'Connection Timeout=30;'
)

def get_db_connection():
    """
    Establishes and returns a connection to the database.
    """
    return pyodbc.connect(CONNECTION_STRING) # pylint: disable=E1101

def get_model(num_classes):
    """
    Loads and prepares the Faster R-CNN model for object detection.
    Args:
        num_classes (int): The number of classes the model should recognize.
    Returns:
        torch.nn.Module: The prepared Faster R-CNN model.
    """
    model_loaded = fasterrcnn_resnet50_fpn(weights='FasterRCNN_ResNet50_FPN_Weights.COCO_V1')
    in_features = model_loaded.roi_heads.box_predictor.cls_score.in_features
    model_loaded.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model_loaded

NUM_CLASSES = 2
model = get_model(NUM_CLASSES)
model.load_state_dict(torch.load(
    'model/bullet_detection_model.pth',
    map_location=torch.device('cpu')
    ))
model.eval()

transform = T.Compose([
    T.ToTensor()
])

def process_image(file):
    """Process the uploaded image and return the image tensor."""
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)
    image = Image.open(filepath).resize((800, 1000))
    return transform(image).unsqueeze(0), filepath

def insert_impact_data(id_usuario, filepath, boxes, scores):
    """Insert impact data into the database."""
    total_precision = 0
    disparos = []
    for box, score in zip(boxes, scores):
        x1, y1, x2, y2 = box
        ubicacion = f'({(x1 + x2) / 2}, {(y1 + y2) / 2})'
        precision = float(score)
        total_precision += precision
        disparos.append({
            'ubicacion': ubicacion,
            'precision': precision
        })
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO ImpactosBala (IdUsuario, Fecha, Ubicacion, Precision, RutaImagen) "
                "VALUES (?, ?, ?, ?, ?)",
                id_usuario, datetime.now(), ubicacion, precision, filepath
            )
            conn.commit()
    return disparos, total_precision

def insert_report_data(id_usuario, total_impactos, promedio_precision):
    """Insert report data into the database."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO Reportes(IdUsuario,FechaReporte,TotalImpactos,PromedioPrecision,Detalles) "
            "VALUES (?, ?, ?, ?, ?)",
            id_usuario,
            datetime.now(),
            total_impactos,
            promedio_precision,
            'Detalles del reporte de impacto de balas.'
        )
        conn.commit()

@app.route('/detect', methods=['POST'])
def detect():
    """
        Endpoint para detectar objetos en una imagen enviada.

        Recibe una imagen y un identificador de usuario a través de una solicitud POST. 
        Guarda la imagen, realiza la detección con un modelo Faster R-CNN, 
        y almacena los resultados en la base de datos SQL. 
        Retorna un JSON con las ubicaciones y precisiones de los impactos detectados.
    """
    file = request.files.get('imageName')
    id_usuario = request.form.get('idUsuario')

    if not file:
        return jsonify({'status': 'error', 'message': 'El campo de imagen es obligatorio'}), 400

    image_tensor, filepath = process_image(file)

    with torch.no_grad():
        predictions = model(image_tensor)

    prediction_data = predictions[0]['boxes'].cpu().numpy()
    scores = predictions[0]['scores'].cpu().numpy()

    threshold = 0.5
    boxes = prediction_data[scores > threshold]
    scores = scores[scores > threshold]

    disparos, total_precision = insert_impact_data(id_usuario, filepath, boxes, scores)
    promedio_precision = total_precision / len(disparos) if disparos else 0
    insert_report_data(id_usuario, len(disparos), promedio_precision)

    return jsonify({'disparos': disparos})


if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(debug=True, host='0.0.0.0', port=5008)

