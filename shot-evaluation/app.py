import os
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
from PIL import Image
import torch
import torchvision.transforms as T
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import pyodbc
from datetime import datetime

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024     

conn_str = (
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
    return pyodbc.connect(conn_str)

def get_model(num_classes):
    model = fasterrcnn_resnet50_fpn(weights='FasterRCNN_ResNet50_FPN_Weights.COCO_V1')
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

num_classes = 2
model = get_model(num_classes)
model.load_state_dict(torch.load('model/bullet_detection_model.pth', map_location=torch.device('cpu')))
model.eval()

transform = T.Compose([
    T.ToTensor()
])

@app.route('/detect', methods=['POST'])
def detect():
    file = request.files.get('imageName')
    id_usuario = request.form.get('idUsuario')

    if not file:
        return jsonify({'status': 'error', 'message': 'El campo de imagen es obligatorio'}), 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    image = Image.open(filepath).resize((800, 1000))
    image_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        predictions = model(image_tensor)

    prediction_data = predictions[0]['boxes'].cpu().numpy()
    scores = predictions[0]['scores'].cpu().numpy()

    threshold = 0.5
    boxes = prediction_data[scores > threshold]
    scores = scores[scores > threshold]

    disparos = []
    total_precision = 0
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
                "INSERT INTO ImpactosBala (IdUsuario, Fecha, Ubicacion, Precision, RutaImagen) VALUES (?, ?, ?, ?, ?)",
                id_usuario, datetime.now(), ubicacion, precision, filepath
            )
            conn.commit()

    promedio_precision = total_precision / len(disparos) if disparos else 0

    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO Reportes (IdUsuario, FechaReporte, TotalImpactos, PromedioPrecision, Detalles) VALUES (?, ?, ?, ?, ?)",
            id_usuario, datetime.now(), len(disparos), promedio_precision, 'Detalles del reporte de impacto de balas.'
        )
        conn.commit()

    return jsonify({'disparos': disparos})

if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(debug=True, host='0.0.0.0', port=5008)
