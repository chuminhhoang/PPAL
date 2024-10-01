import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
from ultralytics import YOLO

class CustomDataset(Dataset):
    def __init__(self, images_folder, labels_folder):
        self.images_folder = images_folder
        self.labels_folder = labels_folder
        self.image_files = [f for f in os.listdir(images_folder) if f.endswith('.jpg')]
        
    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_file = self.image_files[idx]
        label_file = os.path.join(self.labels_folder, img_file.replace('.jpg', '.txt'))

        img_path = os.path.join(self.images_folder, img_file)

        # Đọc bounding boxes và labels
        if os.path.exists(label_file):
            boxes = []
            labels = []
            with open(label_file, 'r') as f:
                for line in f.readlines():
                    parts = list(map(float, line.strip().split()))
                    class_id = int(parts[0])  # Nhãn lớp
                    x_center, y_center, width, height = parts[1:5]

                    # Chuyển đổi từ (x_center, y_center, width, height) sang (x_min, y_min, x_max, y_max)
                    x_min = x_center - width / 2
                    y_min = y_center - height / 2
                    x_max = x_center + width / 2
                    y_max = y_center + height / 2
                    boxes.append([x_min, y_min, x_max, y_max])
                    labels.append(class_id)

            # Chuyển đổi thành tensor
            gt_bboxes = torch.tensor(boxes, dtype=torch.float32)
            gt_labels = torch.tensor(labels, dtype=torch.int64)
        else:
            gt_bboxes = torch.empty((0, 4), dtype=torch.float32)
            gt_labels = torch.empty((0,), dtype=torch.int64)

        return img_path, gt_bboxes, gt_labels

# Hàm để dự đoán với YOLO
def predict_with_yolo(model, image):
    
    results = model(image)  # Dự đoán từ mô hình YOLO
    pred_bboxes = results[0].boxes.xyxyn  # Đầu ra bounding boxes
    pred_cls = results[0].boxes.cls     # Đầu ra lớp
    return pred_bboxes, pred_cls

# Khởi tạo dataset và dataloader
images_folder = '/home/mq/data_disk2T/Thang/bak/src/data1/train/images'
labels_folder = '/home/mq/data_disk2T/Thang/bak/src/data1/train/labels'

dataset = CustomDataset(images_folder, labels_folder)
data_loader = DataLoader(dataset, batch_size=4, shuffle=True)
model = YOLO("/home/mq/data_disk2T/Thang/best.pt")

results = []  # Danh sách lưu trữ kết quả cho tất cả các batch

# Lặp qua các batch dữ liệu
for batch in data_loader:
    images, gt_bboxes_batch, gt_labels_batch = batch

    # Dự đoán với YOLO cho cả batch
    pred_bboxes_batch = []
    pred_cls_batch = []

    for image in images:
        pred_bboxes, pred_cls = predict_with_yolo(model, image)
        pred_bboxes_batch.append(pred_bboxes)
        pred_cls_batch.append(pred_cls)

    # # Chuyển đổi thành tensor
    # pred_bboxes_batch = torch.stack(pred_bboxes_batch)
    # pred_cls_batch = torch.stack(pred_cls_batch)

    # Chuyển đổi sang định dạng có thể lưu trữ
    for i in range(len(images)):
        result = {
            'gt_bboxes': gt_bboxes_batch[i].tolist(),
            'gt_labels': gt_labels_batch[i].tolist(),
            'pred_bboxes': pred_bboxes_batch[i].tolist(),
            'pred_cls': pred_cls_batch[i].tolist()
        }
        results.append(result)

# Lưu kết quả thành tệp JSON
with open('results.json', 'w') as json_file:
    json.dump(results, json_file, indent=4)

print(f"Đã lưu kết quả vào results.json")
