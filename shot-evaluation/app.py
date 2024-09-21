import os
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
from PIL import Image,UnidentifiedImageError
import torch
import torchvision.transforms as T
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import numpy as np
from itertools import combinations
from flask_cors import CORS

app = Flask(__name__)
CORS(app)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

center_target = (680, 1140)
head_center = (680, 300)
heart_center = (680, 700)
img_height = 1920
img_width = 1335

def is_lethal_impact(impact, head_center, heart_center, head_lethal_radius=250, heat_lethal_radius=50):
    is_head_impact = np.linalg.norm(
        np.array(impact) - np.array(head_center)) <= head_lethal_radius
    is_heart_impact = np.linalg.norm(
        np.array(impact) - np.array(heart_center)) <= heat_lethal_radius
    return is_head_impact or is_heart_impact


def calculate_individual_accuracy(impact, center_target, head_center, heart_center):
    if is_lethal_impact(impact, head_center, heart_center):
        return 0.0

    distance_from_center = np.linalg.norm(
        np.array(impact) - np.array(center_target))
    max_distance = np.linalg.norm(
        np.array(center_target) - np.array([0, img_height])) / 1.5
    accuracy = 1 - (distance_from_center / max_distance) ** 2
    return round(accuracy * 100, 2)


def calculate_dispersion(bullet_positions):
    if len(bullet_positions) > 1:
        pairwise_distances = [np.linalg.norm(
            np.array(a) - np.array(b)) for a, b in combinations(bullet_positions, 2)]
        avg_distance_between_bullets = np.mean(pairwise_distances)
        max_bullet_dispersion = np.linalg.norm(
            np.array([0, 0]) - np.array([img_width, img_height]))
        dispersion_factor = 1 - \
            (avg_distance_between_bullets / max_bullet_dispersion)
    else:
        dispersion_factor = 1
    return round(dispersion_factor * 100, 2)


def calculate_final_accuracy(bullet_positions, center_target, head_center, heart_center):
    individual_accuracies = [calculate_individual_accuracy(
        bullet, center_target, head_center, heart_center) for bullet in bullet_positions]
    average_individual_accuracy = sum(
        individual_accuracies) / len(individual_accuracies)
    dispersion_factor = calculate_dispersion(bullet_positions)
    final_accuracy = average_individual_accuracy * 0.8 + dispersion_factor * 0.2
    return round(final_accuracy, 2)


def get_model(num_classes):
    model = fasterrcnn_resnet50_fpn(
        weights='FasterRCNN_ResNet50_FPN_Weights.COCO_V1')
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


num_classes = 2
model = get_model(num_classes)
model.load_state_dict(torch.load(
    'model/bullet_detection_model.pth', map_location=torch.device('cpu')))
model.eval()

transform = T.Compose([T.ToTensor()])


@app.route('/detect', methods=['POST'])
def detect():
    file = request.files.get('image')
    if not file:
        return jsonify({'status': 'error', 'message': 'El campo de imagen es obligatorio'}), 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    try:
        image = Image.open(filepath).resize((img_width, img_height))
    except UnidentifiedImageError:
        return jsonify({'status': 'error', 'message': 'El archivo no es una imagen válida'}), 400

    image_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        predictions = model(image_tensor)

    prediction_data = predictions[0]['boxes'].cpu().numpy()
    scores = predictions[0]['scores'].cpu().numpy()

    threshold = 0.5
    boxes = prediction_data[scores > threshold]
    scores = scores[scores > threshold]

    bullet_positions = []
    disparos = []
    for box, score in zip(boxes, scores):
        x1, y1, x2, y2 = box
        x = (x1 + x2) / 2
        y = (y1 + y2) / 2
        bullet_positions.append((x, y))

        individual_accuracy = calculate_individual_accuracy(
            (x, y), center_target, head_center, heart_center)

        disparos.append({
            'ubicacion': f'({x}, {y})',
            'precision': individual_accuracy
        })
    if len(bullet_positions) == 0:
        return jsonify({'status': 'error', 'message': 'No se detectaron balas'}), 400
 
    final_accuracy = calculate_final_accuracy(
        bullet_positions, center_target, head_center, heart_center)

    return jsonify({
        'disparos': disparos,
        'final_accuracy': final_accuracy
    })


if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(debug=True, host='0.0.0.0', port=5008)


