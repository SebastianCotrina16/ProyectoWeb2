import cv2
import numpy as np
import random
import os
import json
from itertools import combinations

silueta_path = 'plantilla.png'
silueta = cv2.imread(silueta_path)
img_height, img_width, _ = silueta.shape

num_train_images = 8000
num_val_images = 2000
train_output_dir = 'dataset/images/'
val_output_dir = 'dataset/validation/images/'

os.makedirs(train_output_dir, exist_ok=True)
os.makedirs(val_output_dir, exist_ok=True)


def draw_bullet_hole(image, center, radius=10):
    color = (0, 0, 255)
    thickness = -1
    cv2.circle(image, center, radius, color, thickness)


center_target = (680, 1140)
head_center = (680, 300)
heart_center = (680, 700)


def is_lethal_impact(impact, head_center, heart_center, head_lethal_radius=250, heat_lethal_radius=50):
    print(impact, head_center, heart_center,
          np.linalg.norm(np.array(impact) - np.array(head_center)),
          np.linalg.norm(np.array(impact) - np.array(heart_center))
          )

    is_head_impact = np.linalg.norm(
        np.array(impact) - np.array(head_center)) <= head_lethal_radius
    is_heart_impact = np.linalg.norm(
        np.array(impact) - np.array(heart_center)) <= heat_lethal_radius
    if (is_head_impact or is_heart_impact):
        return True
    return False


def calculate_individual_accuracy(impact, center_target, head_center, heart_center):
    """
    Calcula la precisión individual de una bala, penalizando si es un impacto letal (0% si golpea cabeza o corazón).
    """
    if is_lethal_impact(impact, head_center, heart_center):
        return 0.0

    distance_from_center = np.linalg.norm(
        np.array(impact) - np.array(center_target))

    max_distance = np.linalg.norm(
        np.array(center_target) - np.array([0, img_height])) / 1.5

    accuracy = 1 - (distance_from_center / max_distance) ** 2
    return round(accuracy * 100, 2)


def calculate_dispersion(bullet_positions):
    """
    Calcula la dispersión entre las balas: distancia promedio entre las posiciones de impacto.
    """
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
    """
    Calcula la precisión general basada en las precisiones individuales y la dispersión entre balas.
    """
    individual_accuracies = [calculate_individual_accuracy(
        bullet, center_target, head_center, heart_center) for bullet in bullet_positions]

    average_individual_accuracy = sum(
        individual_accuracies) / len(individual_accuracies)

    dispersion_factor = calculate_dispersion(bullet_positions)

    final_accuracy = average_individual_accuracy * 0.8 + \
        dispersion_factor * 0.2
    return round(final_accuracy, 2)


def generate_images_and_annotations(num_images, output_dir, annotation_file):
    annotations = []
    for i in range(num_images):
        img = silueta.copy()

        annotation = {
            "image": f"image_{i+1}.jpg",
            "annotations": []
        }

        num_bullet_holes = random.randint(3, 7)
        bullet_positions = []
        print(i+1)
        for _ in range(num_bullet_holes):
            x = random.randint(150, 1250)
            y = random.randint(5, 1900)
            bullet_positions.append((x, y))
            draw_bullet_hole(img, (x, y))

            bbox = [x - 10, y - 10, x + 10, y + 10]
            individual_accuracy = calculate_individual_accuracy(
                (x, y), center_target, head_center, heart_center)

            annotation["annotations"].append({
                "class": "bullet_hole",
                "bbox": bbox,
                "individual_accuracy": individual_accuracy
            })

        final_accuracy = calculate_final_accuracy(
            bullet_positions, center_target, head_center, heart_center)
        annotation["final_accuracy"] = final_accuracy

        annotations.append(annotation)
        output_path = os.path.join(output_dir, f'image_{i+1}.jpg')
        cv2.imwrite(output_path, img)

    with open(annotation_file, 'w') as f:
        json.dump(annotations, f)


generate_images_and_annotations(
    num_train_images, train_output_dir, 'dataset/annotations.json')
generate_images_and_annotations(
    num_val_images, val_output_dir, 'dataset/validation/annotations.json')

print("Imágenes y anotaciones generadas.")

