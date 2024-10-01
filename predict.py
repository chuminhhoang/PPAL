import os
from ultralytics import YOLO
from torch.utils.data import Dataset, DataLoader
images_folder = '/home/mq/data_disk2T/Thang/bak/src/data1/train/images'
# image_files = [f for f in os.listdir(images_folder) if f.endswith('.jpg')]
# img_file = image_files[4]
# img_path = os.path.join(images_folder, img_file)
# # Load a model
# model = YOLO("/home/mq/data_disk2T/Thang/runs/detect/train6/weights/best.pt")
# results = model(img_path)  # Dự đoán từ mô hình YOLO
# pred_bboxes = results[0].boxes.xyxyn  # Đầu ra bounding boxes
# pred_cls = results[0].boxes.cls     # Đầu ra lớp
# pred_conf=results[0].boxes.conf
# print(pred_bboxes)
# print(pred_cls)
# print(pred_conf)
class CustomImageDataset(Dataset):
    def __init__(self, image_folder, transform=None):
        self.image_folder = image_folder
        self.transform = transform
        self.image_files = [f for f in os.listdir(image_folder) if f.endswith(('jpg', 'png', 'jpeg'))]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_folder, self.image_files[idx])
        return img_path

# Sử dụng CustomImageDataset
custom_dataset = CustomImageDataset(images_folder)
data_loader = DataLoader(custom_dataset, batch_size=4, shuffle=True)
# Lặp qua các batch dữ liệu
for batch in data_loader:
    print(batch)
    break
