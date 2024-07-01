import cv2
import numpy as np
import random
import os
import json

img_width = 800
img_height = 1000
num_train_images = 150
num_val_images = 50
train_output_dir = 'dataset/images/'
val_output_dir = 'dataset/validation/images/'

os.makedirs(train_output_dir, exist_ok=True)
os.makedirs(val_output_dir, exist_ok=True)

def draw_bullet_hole(image, center, radius=5):
    color = (0, 0, 255)  
    thickness = -1 
    cv2.circle(image, center, radius, color, thickness)

def generate_images_and_annotations(num_images, output_dir, annotation_file):
    annotations = []
    for i in range(num_images):
        img = np.ones((img_height, img_width, 3), dtype=np.uint8) * 255
        center = (img_width // 2, img_height // 2)
        radius = 150
        cv2.circle(img, center, radius, (0, 0, 0), 2)
        
        annotation = {
            "image": f"image_{i+1}.jpg",
            "annotations": []
        }

        num_bullet_holes = random.randint(3, 10)
        for _ in range(num_bullet_holes):
            x = random.randint(center[0] - radius, center[0] + radius)
            y = random.randint(center[1] - radius, center[1] + radius)
            draw_bullet_hole(img, (x, y))
            bbox = [x - 5, y - 5, x + 5, y + 5]
            annotation["annotations"].append({
                "class": "bullet_hole",
                "bbox": bbox
            })

        annotations.append(annotation)
        output_path = os.path.join(output_dir, f'image_{i+1}.jpg')
        cv2.imwrite(output_path, img)

    with open(annotation_file, 'w') as f:
        json.dump(annotations, f)

generate_images_and_annotations(num_train_images, train_output_dir, 'dataset/annotations.json')
generate_images_and_annotations(num_val_images, val_output_dir, 'dataset/validation/annotations.json')

print("Imágenes y anotaciones generadas.")
