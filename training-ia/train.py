import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor, FasterRCNN_ResNet50_FPN_Weights
from torchvision.transforms import functional as F
from torch.utils.data import DataLoader, Dataset
import os
import json
from PIL import Image
from torchvision import transforms as T
print(torch.cuda.is_available())

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
    transforms.append(T.ToTensor())
    transforms.append(T.RandomHorizontalFlip(0.5))
    transforms.append(T.ColorJitter(brightness=0.2, contrast=0.2,
                      saturation=0.2, hue=0.1))
    return T.Compose(transforms)


def get_model(num_classes):
    weights = FasterRCNN_ResNet50_FPN_Weights.COCO_V1
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
        weights=weights)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


def collate_fn(batch):
    return tuple(zip(*batch))


def train_model():
    dataset = BulletDataset(
        "dataset/images", "dataset/annotations.json", get_transform())
    dataset_test = BulletDataset(
        "dataset/validation/images", "dataset/validation/annotations.json", get_transform())

    data_loader = DataLoader(
        dataset, batch_size=2, shuffle=True, num_workers=2, collate_fn=collate_fn)
    data_loader_test = DataLoader(
        dataset_test, batch_size=2, shuffle=False, num_workers=2, collate_fn=collate_fn)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    num_classes = 2
    model = get_model(num_classes)
    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params, lr=0.005, momentum=0.9, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=3, gamma=0.1)

    num_epochs = 10

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for i, (images, targets) in enumerate(data_loader):
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()}
                       for t in targets]

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            running_loss += losses.item()

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            if i % 10 == 0:
                print(
                    f"Epoch [{epoch+1}/{num_epochs}], Step [{i}/{len(data_loader)}], Loss: {losses.item():.4f}")

        lr_scheduler.step()

        print(
            f"Epoch [{epoch+1}/{num_epochs}] finished with average loss: {running_loss/len(data_loader):.4f}")

        model.eval()
        print(f"Starting evaluation for epoch [{epoch+1}/{num_epochs}]")
        with torch.no_grad():
            for i, (images, targets) in enumerate(data_loader_test):
                images = list(image.to(device) for image in images)
                targets = [{k: v.to(device) for k, v in t.items()}
                           for t in targets]
                outputs = model(images)

                if i % 10 == 0:
                    print(
                        f"Validation Step [{i}/{len(data_loader_test)}] completed")

        print(f"Evaluation for epoch [{epoch+1}/{num_epochs}] completed")

    torch.save(model.state_dict(), 'bullet_detection_model.pth')
    print("Entrenamiento completado")


if __name__ == '__main__':
    train_model()
