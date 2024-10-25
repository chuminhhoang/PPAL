from bboxes_iou import bbox_overlaps
from ultra.ultralytics.models.yolo.model import YOLO
import torch
from quality import Class_Quality
import json
import shutil
import glob
from get_pos_mask import get_pos_mask
from ultra.ultralytics.nn import DetectionModel
import torch
import cv2
import numpy as np
from torchvision.ops import nms
from ultra.ultralytics.utils.tal import make_anchors
from ultra.ultralytics.utils.tal import TaskAlignedAssigner
from dataloader import DetectionTrainer
from utils import get_img_score_distance_matrix_slow

max_bbox =  200

def dist2bbox(distance, anchor_points, xywh=True, dim=-1):
    """Transform distance(ltrb) to box(xywh or xyxy)."""
 
    lt, rb = distance.chunk(2, dim)
    x1y1 = anchor_points - lt
    x2y2 = anchor_points + rb
    if xywh:
        c_xy = (x1y1 + x2y2) / 2
        wh = x2y2 - x1y1
        return torch.cat((c_xy, wh), dim)  # xywh bbox
    return torch.cat((x1y1, x2y2), dim)  # xyxy bbox

def bbox_decode(anchor_points, pred_dist):
            proj=torch.arange(16, dtype=torch.float).cpu()
            """Decode predicted object bounding box coordinates from anchor points and distribution."""
            b, a, c = pred_dist.shape  # batch, anchors, channels
            pred_dist = pred_dist.view(b, a, 4, c // 4).softmax(3).matmul(proj.type(pred_dist.dtype))
 
            # pred_dist = pred_dist.view(b, a, c // 4, 4).transpose(2,3).softmax(3).matmul(self.proj.type(pred_dist.dtype))
            # pred_dist = (pred_dist.view(b, a, c // 4, 4).softmax(2) * self.proj.type(pred_dist.dtype).view(1, 1, -1, 1)).sum(2)
            return dist2bbox(pred_dist, anchor_points, xywh=False)

def pre(img0):
    img0, w, h, width, height = letterbox(img0)
    img0 = cv2.cvtColor(img0, cv2.COLOR_BGR2RGB)
    img0 = img0 / 255.0
    img0 = img0.transpose(2, 0, 1)
    return img0, w, h, width, height

def letterbox(img, new_shape = (640, 640), color = (114, 114, 114), 
              auto = False, scale_fill = False, scaleup = False, stride = 32):
    
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

def predict_img(model, image_path, diversity):
    batch_images = []
    for img_path in [image_path]:
         img=cv2.imread(img_path)
         x, w, h, width, height = pre(img)
         x = torch.from_numpy(x).to('cpu').float()  # Chuyển đổi từ numpy thành tensor
         batch_images.append(x)
    batch_images = torch.stack(batch_images)
    with torch.no_grad():
        preds = model.predict(batch_images, diversity = diversity)
        
    return preds

def extract_feature(model, image_path, num_classes):

    pred_feats = predict_img(model, image_path, diversity = True).transpose(1, 2)
    preds = predict_img(model, image_path, diversity = False)
    feats = preds[1] if isinstance(preds, tuple) else preds
    m=model.model[-1]
    nc=num_classes
    no= nc + m.reg_max * 4
    reg_max=m.reg_max
    stride=m.stride
    pred_distri, pred_scores = torch.cat([xi.view(feats[0].shape[0], no, -1) for xi in feats], 2).split(
            (reg_max * 4, nc), 1
        )
    pred_scores = pred_scores.permute(0, 2, 1).contiguous().sigmoid()
  
    pred_distri = pred_distri.permute(0, 2, 1).contiguous()
    dtype = pred_scores.dtype
    batch_size = pred_scores.shape[0]
    # imgsz = torch.tensor(feats[0].shape[2:],  dtype=dtype) * stride[0].to('cpu') # image size (h,w)
    anchor_points, stride_tensor = make_anchors(feats, stride, 0.5)
    # anc_points=anchor_points * stride_tensor
    
    # Targets
    pred_bboxes = bbox_decode(anchor_points, pred_distri)*stride_tensor 

    mlvl_labels = []
    mlvl_scores = []
    for idx, pred_bbox in enumerate(pred_bboxes):
        max_values, label = torch.max(pred_scores[idx], dim=1)
        mlvl_labels.append(label.cpu())
        mlvl_scores.append(max_values.cpu())
        keep_ids = nms(pred_bbox, max_values, 0.3).cpu()

    mlvl_labels = torch.from_numpy(np.array(mlvl_labels[0], dtype=np.float32))
    mlvl_scores = torch.from_numpy(np.array(mlvl_scores[0], dtype=np.float32))

    det_labels = mlvl_labels[keep_ids][:max_bbox]
    det_scores = mlvl_scores[keep_ids][:max_bbox]
    det_bboxes = pred_bbox[keep_ids][:max_bbox]
    det_feats = pred_feats[0][keep_ids][:max_bbox]


    return det_labels, det_bboxes, det_feats, det_scores


a = YOLO('/home/mq/data_disk2T/Thang/bak/src/runs/detect/train10/weights/best.pt')
model  = a.model
# print(model)
model.to('cpu')


def get_all_meta_data(list_img_path, feat_dim):

    queue_det_feats = torch.zeros((len(list_img_path), 200, feat_dim))
    queue_det_labels = torch.zeros((len(list_img_path), 200))
    queue_det_scores = torch.zeros((len(list_img_path), 200))
    queue_det_idx = torch.zeros((len(list_img_path), 1))

    for idx, img_path in enumerate(list_img_path):
        det_labels, det_bboxes, det_feats, det_scores = extract_feature(model, img_path, 3) 
        queue_det_idx[idx] = int(idx)
        queue_det_feats[idx] = det_feats
        queue_det_labels[idx] = det_labels
        queue_det_scores[idx] = det_scores

    return queue_det_idx, queue_det_feats, queue_det_labels, queue_det_scores

def compute_al(valid_inds, list_path, output_path, output_txt):
    
    list_img_uncertainty_path = []
    for i in valid_inds:
        list_img_uncertainty_path.append(list_path[i])

    queue_det_idx, queue_det_feats, queue_det_labels, queue_det_scores = get_all_meta_data(list_img_uncertainty_path, 64)
    # queue_det_idx = queue_det_idx[valid_inds]
    # queue_det_feats = queue_det_feats[valid_inds]
    # queue_det_labels = queue_det_labels[valid_inds]
    # queue_det_scores = queue_det_scores[valid_inds]
    
    img_dis_mat = get_img_score_distance_matrix_slow(
            queue_det_labels, queue_det_scores, queue_det_feats, score_thr=0.05)
    
    img_dis_mat = img_dis_mat.detach().cpu().numpy()
    img_ids = queue_det_idx.detach().cpu().numpy()
    with open(output_path, 'wb') as fwb:
            np.save(fwb, img_dis_mat)
            np.save(fwb, img_ids)

    with open(output_txt, 'w') as f:
            for line in list_img_uncertainty_path:
                f.write(f"{line}\n")
    return

a = glob.glob('/home/mq/data_disk2T/Thang/bak/src/data1/val/images/*.jpg')

compute_al([190, 2, 264, 367, 425, 13, 62, 219, 183, 443, 57, 419, 248, 131, 432, 316, 281, 485, 147, 280, 487, 396, 206, 10, 456, 382, 120, 295, 263, 209], a, 'new.npy', 'out.txt')