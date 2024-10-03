import torch
import json

from get_pos_mask import get_pos_mask
from ultralytics.nn.tasks import DetectionModel
import torch
import cv2
import numpy as np
# from get_fg_mask import select_ghest_overlaps
from iou import iou
def letterbox(img, new_shape = (640, 640), color = (114, 114, 114), 
              auto = False, scale_fill = False, scaleup = False, stride = 32):
    
    # Resize and pad image while meeting stride-multiple constraints
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scale_fill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    height, width = img.shape[:2]
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return img, dw, dh, width, height

def pre(img0):
    img0, w, h, width, height = letterbox(img0)
    img0 = cv2.cvtColor(img0, cv2.COLOR_BGR2RGB)
    img0 = img0 / 255.0
    img0 = img0.transpose(2, 0, 1)
    return img0, w, h, width, height
def predict_img(image_path):
    model  = DetectionModel()
    model.load(torch.load('E:\\4.MQ_ICT\PPAL\yolov8n.pt'))
    batch_images = []
    for img_path in image_path:
         img=cv2.imread(img_path)
         x, w, h, width, height = pre(img)
         x = torch.from_numpy(x).float()  # Chuyển đổi từ numpy thành tensor
         batch_images.append(x)
    batch_images = torch.stack(batch_images)
    with torch.no_grad():
        preds = model.forward(batch_images)
    return preds

# Đường dẫn đến file JSON chứa dữ liệu
json_file_path = 'output_data.json'  # Thay đổi đường dẫn này nếu cần

# Đọc dữ liệu từ file JSON
with open(json_file_path, 'r') as file:
    data = json.load(file)

# Lọc dữ liệu theo batch_idx = 1
filtered_data = [item for item in data if item["batch_idx"] == 1]

# Chuyển đổi dữ liệu thành tensor
batch_size = 5  # Kích thước của batch
image_path={item['file_name'] for item in filtered_data}
preds=predict_img(image_path)
feats = preds[1] if isinstance(preds, tuple) else preds
model  = DetectionModel()
model.load(torch.load('E:\\4.MQ_ICT\PPAL\yolov8n.pt'))
m=model.model[-1]
nc=3
no= nc + m.reg_max
reg_max=m.reg_max
stride=m.stride
pred_distri, pred_scores = torch.cat([xi.view(feats[0].shape[0], no, -1) for xi in feats], 2).split(
            (reg_max * 4, nc), 1
        )
# print(pred_distri)
# exit()
pred_scores = pred_scores.permute(0, 2, 1).contiguous()
pred_distri = pred_distri.permute(0, 2, 1).contiguous()

dtype = pred_scores.dtype
batch_size = pred_scores.shape[0]
imgsz = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]  # image size (h,w)
anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)
image_idx = torch.tensor([item["image_idx"] for item in filtered_data])
labels = torch.tensor([item["labels"] for item in filtered_data])
bboxes = torch.tensor([item["bbox"] for item in filtered_data])
# Kết hợp các tensor lại với nhau
targets = torch.cat((image_idx.view(-1, 1), labels.view(-1, 1), bboxes), 1)
print(targets)
nl, ne = targets.shape
i = targets[:, 0]  # image index
_, counts = i.unique(return_counts=True)
counts = counts.to(dtype=torch.int32)
out = torch.zeros(batch_size, counts.max(), ne - 1)
for j in range(batch_size):
                matches = i == j
                n = matches.sum()
                if n:
                    out[j, :n] = targets[matches, 1:]
gt_labels, gt_bboxes = out.split((1, 4), 2)  # cls, xyxy
mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)
print(gt_bboxes.shape)
# # Tách thành gt_labels và gt_bboxes
# gt_labels, gt_bboxes = targets.split((1, 4), 1)  # cls, xyxy

# # Xuất kết quả
# print("Ground Truth Labels:", gt_labels)
# print("Ground Truth Bboxes:", gt_bboxes)
