import os
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
from PIL import Image
import torch
import torchvision.transforms as T
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB

def get_model(num_classes):
    model = fasterrcnn_resnet50_fpn(pretrained=True)
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
    for box, score in zip(boxes, scores):
        x1, y1, x2, y2 = box
        disparos.append({
            'ubicacion': f'({(x1 + x2) / 2}, {(y1 + y2) / 2})',
            'precision': float(score)
        })

    return jsonify({'disparos': disparos})

if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(debug=True, host='0.0.0.0', port=5000)
