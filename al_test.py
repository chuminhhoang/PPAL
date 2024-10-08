from bboxes_iou import bbox_overlaps
from ultralytics import YOLO
import torch
from quality import Class_Quality
import json
import shutil
from get_pos_mask import get_pos_mask
from ultralytics.nn.tasks import DetectionModel
import torch
import cv2
import numpy as np
from ultralytics.utils.tal import make_anchors
from ultralytics.utils.tal import TaskAlignedAssigner
from dataloader import DetectionTrainer
from ultralytics.hub import HUB_WEB_ROOT, HUBTrainingSession
from ultralytics.utils import (
    ARGV,
    ASSETS,
    DEFAULT_CFG_DICT,
    LOGGER,
    RANK,
    SETTINGS,
    callbacks,
    checks,
    emojis,
    yaml_load,
)
from ultralytics.utils import (
    DEFAULT_CFG,
    LOCAL_RANK,
    LOGGER,
    RANK,
    TQDM,
    __version__,
    callbacks,
    clean_url,
    colorstr,
    emojis,
    yaml_save,
)
# from get_fg_mask import select_ghest_overlaps
from calculator import bbox_iou
def xywh2xyxy(x):
    """
    Convert bounding box coordinates from (x, y, width, height) format to (x1, y1, x2, y2) format where (x1, y1) is the
    top-left corner and (x2, y2) is the bottom-right corner. Note: ops per 2 channels faster than per channel.

    Args:
        x (np.ndarray | torch.Tensor): The input bounding box coordinates in (x, y, width, height) format.

    Returns:
        y (np.ndarray | torch.Tensor): The bounding box coordinates in (x1, y1, x2, y2) format.
    """
    assert x.shape[-1] == 4, f"input shape last dimension expected 4 but input shape is {x.shape}"
    y = torch.empty_like(x) if isinstance(x, torch.Tensor) else np.empty_like(x)  # faster than clone/copy
    xy = x[..., :2]  # centers
    wh = x[..., 2:] / 2  # half width-height
    y[..., :2] = xy - wh  # top left xy
    y[..., 2:] = xy + wh  # bottom right xy
    return y
def preprocess(targets, batch_size, scale_tensor):
        """Preprocesses the target counts and matches with the input batch size to output a tensor."""
        nl, ne = targets.shape
        if nl == 0:
            out = torch.zeros(batch_size, 0, ne - 1)
        else:
            i = targets[:, 0]  # image index
            _, counts = i.unique(return_counts=True)
            counts = counts.to(dtype=torch.int32)
            out = torch.zeros(batch_size, counts.max(), ne - 1)
            for j in range(batch_size):
                matches = i == j
                n = matches.sum()
                if n:
                    out[j, :n] = targets[matches, 1:]
            out[..., 1:5] = xywh2xyxy(out[..., 1:5].mul_(scale_tensor))
        return out

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
def bbox_decode( anchor_points, pred_dist):
            proj=torch.arange(16, dtype=torch.float)
            """Decode predicted object bounding box coordinates from anchor points and distribution."""
            b, a, c = pred_dist.shape  # batch, anchors, channels
            pred_dist = pred_dist.view(b, a, 4, c // 4).softmax(3).matmul(proj.type(pred_dist.dtype))
            # pred_dist = pred_dist.view(b, a, c // 4, 4).transpose(2,3).softmax(3).matmul(self.proj.type(pred_dist.dtype))
            # pred_dist = (pred_dist.view(b, a, c // 4, 4).softmax(2) * self.proj.type(pred_dist.dtype).view(1, 1, -1, 1)).sum(2)
            return dist2bbox(pred_dist, anchor_points, xywh=False)
def select_highest_overlaps(mask_pos, overlaps, n_max_boxes):
        """
        Select anchor boxes with highest IoU when assigned to multiple ground truths.

        Args:
            mask_pos (torch.Tensor): Positive mask, shape (b, n_max_boxes, h*w).
            overlaps (torch.Tensor): IoU overlaps, shape (b, n_max_boxes, h*w).
            n_max_boxes (int): Maximum number of ground truth boxes.

        Returns:
            target_gt_idx (torch.Tensor): Indices of assigned ground truths, shape (b, h*w).
            fg_mask (torch.Tensor): Foreground mask, shape (b, h*w).
            mask_pos (torch.Tensor): Updated positive mask, shape (b, n_max_boxes, h*w).

        Note:
            b: batch size, h: height, w: width.
        """
        # Convert (b, n_max_boxes, h*w) -> (b, h*w)
        fg_mask = mask_pos.sum(-2)
        if fg_mask.max() > 1:  # one anchor is assigned to multiple gt_bboxes
            mask_multi_gts = (fg_mask.unsqueeze(1) > 1).expand(-1, n_max_boxes, -1)  # (b, n_max_boxes, h*w)
            max_overlaps_idx = overlaps.argmax(1)  # (b, h*w)

            is_max_overlaps = torch.zeros(mask_pos.shape, dtype=mask_pos.dtype, device=mask_pos.device)
            is_max_overlaps.scatter_(1, max_overlaps_idx.unsqueeze(1), 1)

            mask_pos = torch.where(mask_multi_gts, is_max_overlaps, mask_pos).float()  # (b, n_max_boxes, h*w)
            fg_mask = mask_pos.sum(-2)
        # Find each grid serve which gt(index)
        target_gt_idx = mask_pos.argmax(-2)  # (b, h*w)
        return target_gt_idx, fg_mask, mask_pos
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
    model.load(torch.load('/home/mq/data_disk2T/Thang/best.pt'))
    batch_images = []
    for img_path in image_path:
         img=cv2.imread(img_path)
         x, w, h, width, height = pre(img)
         x = torch.from_numpy(x).float()  # Chuyển đổi từ numpy thành tensor
         batch_images.append(x)
    batch_images = torch.stack(batch_images)
    with torch.no_grad():
        preds = model(batch_images)
    return preds

# tính loss cho từng ảnh 
def single_loss(model, batch, num_classes, quality_xi=0.6):
    image_path = batch['im_file']
    preds=predict_img(image_path)
    feats = preds[1] if isinstance(preds, tuple) else preds
    m=model.model[-1]
    nc=num_classes
    no= nc + m.reg_max * 4
    reg_max=m.reg_max
    stride=m.stride
    # print(feats[0].view(feats[0].shape[0], no, -1).shape)
    pred_distri, pred_scores = torch.cat([xi.view(feats[0].shape[0], no, -1) for xi in feats], 2).split(
            (reg_max * 4, nc), 1
        )
    pred_scores = pred_scores.permute(0, 2, 1).contiguous()
    pred_distri = pred_distri.permute(0, 2, 1).contiguous()
    
    dtype = pred_scores.dtype
    batch_size = pred_scores.shape[0]
    imgsz = torch.tensor(feats[0].shape[2:],  dtype=dtype) * stride[0]  # image size (h,w)
    anchor_points, stride_tensor = make_anchors(feats, stride, 0.5)
    
    # print(anchor_points.shape)
    # exit()
    anc_points=anchor_points * stride_tensor
    
    # Targets
    targets = torch.cat((batch["batch_idx"].view(-1, 1), batch["cls"].view(-1, 1), batch["bboxes"]), 1)
    targets = preprocess(targets, batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
    gt_labels, gt_bboxes = targets.split((1, 4), 2)  # cls, xyxy
    mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)
    pred_bboxes = bbox_decode(anchor_points, pred_distri)
    
    mask_pos, align_metric, overlaps = get_pos_mask(
            pred_scores, pred_bboxes, gt_labels, gt_bboxes, anc_points, mask_gt
        )
    
    n_max_boxes = gt_bboxes.shape[1]
    target_gt_idx, fg_mask, mask_pos = select_highest_overlaps(mask_pos, overlaps, n_max_boxes)
    assigner=TaskAlignedAssigner(topk=10, num_classes=3, alpha=0.5, beta=6.0)
    _, target_bboxes, target_scores, fg_mask, target_gt_idx = assigner(
            pred_scores.detach().sigmoid(),
            (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt,
        )
    
    fg_mask=fg_mask.bool()
    # Lấy chỉ số của các giá trị khác 0
    indices = target_scores[fg_mask].nonzero()
    # Tạo tensor chứa chỉ số cột tương ứng với các giá trị khác 0
    values = target_scores[fg_mask][indices[:, 0], indices[:, 1]]

    target_bboxes /= stride_tensor
    iou=bbox_iou(pred_bboxes[fg_mask], target_bboxes[fg_mask], xywh=False, CIoU=True)
    _labels = torch.argmax(target_scores[fg_mask], dim=1)
    p = values
    quality =1- torch.pow(p, quality_xi) * torch.pow(iou.squeeze(-1), 1. - quality_xi)

    classwise_quality = torch.stack((_labels, quality), dim=-1)
    return classwise_quality

def loss(model, batch, class_quality, class_momentum, num_classes,  base_momentum=0.999):
    classwise_quality =single_loss(model, batch, num_classes)
    with torch.no_grad():
            # classwise_quality = torch.cat(classwise_quality, dim=0)
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
            class_quality = class_momentum * class_quality + \
                        (1. - class_momentum) * avg_qualities
            class_momentum = torch.where(
                avg_qualities > 0,
                torch.zeros_like(class_momentum) + base_momentum,
                class_momentum * base_momentum)
    return class_quality, class_momentum  
def run():
    num_classes=3
    base_momentum = 0.999
    class_quality = torch.zeros((num_classes,))
    class_momentum=torch.ones((num_classes,)) * base_momentum
    a = YOLO('/home/mq/data_disk2T/Thang/best.pt')
    train_loader = a.return_dataset(data = '/home/mq/data_disk2T/Thang/bak/src/data1/data.yaml')
    model  = DetectionModel()
    model.load(torch.load('/home/mq/data_disk2T/Thang/best.pt'))
    for i, batch in train_loader:
        class_quality, class_momentum = loss(model, batch, class_quality, class_momentum, num_classes)
        break
    return class_quality

class_quality = run()
data = {
    'class_quality': class_quality.tolist(),  # Chuyển tensor về list nếu là Tensor
}
# Lưu vào file JSON
with open('/home/mq/data_disk2T/Thang/MTagi/class_quality.json', 'w') as f:
    json.dump(data, f)
   


        
        
        
    