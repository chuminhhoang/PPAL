from bboxes_iou import bbox_overlaps
from ultralytics import YOLO
import torch
from quality import Class_Quality
import os
import shutil
from get_mask_gt import preprocess
from get_pos_mask import get_pos_mask
# from get_fg_mask import select_highest_overlaps
from iou import iou


def load_bboxes_gt(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    bbox_list = []
    for line in lines:
        parts = line.strip().split() 
        bbox = torch.tensor([float(part) for part in parts[1:]])  # Chuyển đổi sang float
        bbox_list.append(bbox)

    bboxes_gt = torch.stack(bbox_list)

    return bboxes_gt
def load_bboxes_pred(file_path):
    # Load a model
    # init Yolo_new_head (Uncertainty) in here after load pretrained weight 
    model = YOLO("/home/mq/data_disk2T/Thang/runs/detect/train6/weights/best.pt")
    # Chạy dự đoán
    results = model.predict(source=file_path, save=False, show=True)
    return results[0].boxes.xyxy, results[0].boxes.conf, results[0].boxes.cls

# tính loss cho từng ảnh 
def single_loss(preds, batch, device, nc, stride, reg_max=16, quality_xi=0.6):
    no=nc+reg_max*4
    feats = preds[1] if isinstance(preds, tuple) else preds
    batch_size=feats[0].shape[0]
    pred_bboxes , pred_scores = torch.cat([xi.view(batch_size, no, -1) for xi in feats], 2).split(
            (reg_max * 4, nc), 1
        )
    pred_scores = pred_scores.permute(0, 2, 1).contiguous()
    dtype = pred_scores.dtype
    imgsz = torch.tensor(feats[0].shape[2:], device=device, dtype=dtype) * stride[0]  # image size (h,w)
    targets = torch.cat((batch["batch_idx"].view(-1, 1), batch["cls"].view(-1, 1), batch["bboxes"]), 1)
    targets = preprocess(targets.to(device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
    gt_labels , gt_bboxes = targets.split((1, 4), 2)  # cls, xyxy
    mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)
    mask_pos, align_metric, overlaps = get_pos_mask(
            pred_scores, pred_bboxes, gt_labels, gt_bboxes, anc_points, mask_gt
        )
    target_gt_idx, fg_mask, mask_pos = select_highest_overlaps(mask_pos, overlaps, n_max_boxes)
    iou= iou(pred_bboxes, gt_bboxes, fg_mask)
    #đưa vào từng bath ảnh 
    classwise_quality_list=[]
    #     bboxes_pred , p, _labels=load_bboxes_pred(file)
    #     bboxes_gt= load_bboxes_gt(file)
    #     quality = torch.pow(p, quality_xi) * torch.pow(iou, 1. - quality_xi)
    #     classwise_quality = torch.stack((_labels, quality), dim=-1) 
    #     classwise_quality_list.append(classwise_quality)
    quality=torch.stack(classwise_quality_list)
    return quality
def loss(bath, num_classes, base_momentum=0.999):
    classwise_quality=single_loss(bath)
    with torch.no_grad():
            classwise_quality = torch.cat(classwise_quality, dim=0)
            _classes = classwise_quality[:, 0]
            _qualities = classwise_quality[:, 1]

            collected_counts = classwise_quality.new_full((num_classes,), 0)
            collected_qualities = classwise_quality.new_full((num_classes,), 0)
            for i in range(num_classes):
                cq = _qualities[_classes == i]
                if cq.numel() > 0:
                    collected_counts[i] += torch.ones_like(cq).sum()
                    collected_qualities[i] += cq.sum()
            avg_qualities = collected_qualities / (collected_counts + 1e-5)
            quality=Class_Quality(num_classes, base_momentum)
            quality.load_from_file()
        
            quality.class_quality = quality.class_momentum * quality.class_quality + \
                        (1. - quality.class_momentum) * avg_qualities
            quality.class_momentum = torch.where(
                avg_qualities > 0,
                torch.zeros_like(quality.class_momentum) + quality.base_momentum,
                quality.class_momentum * quality.base_momentum)
            quality.save_to_file()
            
def run():
    batch=""
    preds=""

    
    


        
        
        
    