import torch
from torch.utils.data import DataLoader, Dataset
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torchvision.transforms import functional as F
from PIL import Image
import os
import json
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

class BulletDataset(Dataset):
    def __init__(self, root, annotation_file, transforms=None):
        self.root = root
        self.transforms = transforms
        with open(annotation_file) as f:
            self.annotations = json.load(f)

    def __getitem__(self, idx):
        annotation = self.annotations[idx]
        img_path = os.path.join(self.root, annotation['image'])
        img = Image.open(img_path).convert("RGB")

        boxes = []
        labels = []
        for ann in annotation['annotations']:
            bbox = ann['bbox']
            boxes.append(bbox)
            labels.append(1)

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)

        target = {"boxes": boxes, "labels": labels}

        if self.transforms is not None:
            img = self.transforms(img)

        return img, target

    def __len__(self):
        return len(self.annotations)


def get_transform():
    transforms = []
    transforms.append(F.to_tensor)
    return torchvision.transforms.Compose(transforms)


def get_model(num_classes):
    weights = torchvision.models.detection.FasterRCNN_ResNet50_FPN_Weights.COCO_V1
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
        weights=weights)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


def load_model(model_path, num_classes, device):
    model = get_model(num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model


def evaluate_model(model, data_loader_test, device):
    metric = MeanAveragePrecision(iou_thresholds=[0.5])

    with torch.no_grad():
        for images, targets in data_loader_test:
            images = list(image.to(device) for image in images)
            outputs = model(images)

            for i in range(len(outputs)):
                preds = {
                    'boxes': outputs[i]['boxes'].cpu(),
                    'scores': outputs[i]['scores'].cpu(),
                    'labels': outputs[i]['labels'].cpu()
                }
                target = {
                    'boxes': targets[i]['boxes'].cpu(),
                    'labels': targets[i]['labels'].cpu()
                }

                metric.update([preds], [target])

    final_map = metric.compute()
    print(f"mAP: {final_map['map']:.4f}")


def collate_fn(batch):
    return tuple(zip(*batch))


if __name__ == "__main__":
    model_path = 'bullet_detection_model.pth'
    data_root = "dataset/validation/images"
    annotation_file = "dataset/validation/annotations.json"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    num_classes = 2

    dataset_test = BulletDataset(data_root, annotation_file, get_transform())
    data_loader_test = DataLoader(
        dataset_test, batch_size=2, shuffle=False, num_workers=2, collate_fn=collate_fn)
    model = load_model(model_path, num_classes, device)
    evaluate_model(model, data_loader_test, device)
