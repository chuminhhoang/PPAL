import os
import cv2
import glob
import json
import torch 
from ultra.ultralytics.models.yolo.model import YOLO
from collections import OrderedDict
# from al_quality import load_bboxes_pred
import json
import numpy as np
import os


def __init__model__(pretrain_weight = ''):
    
    model = YOLO(pretrain_weight).to('cpu')
    return model




def create_json_inference(batch_bboxes, batch_cls_scores , batch_labels, batch_cls_uncertainty, info_meta, json_file):
    
    if not os.path.exists(json_file):        
        json_content = []
    else:
        with open(json_file, 'r') as f:
            json_content = json.load(f)
    for idx, bboxes in enumerate(batch_bboxes):
        for ids, each_bbox in enumerate(bboxes):
            item = {
                'bbox' : each_bbox.tolist(),
                'category_id' : int(batch_labels[idx][ids]),
                'cls_uncertainty': batch_cls_uncertainty[idx][ids].tolist(),
                'file_name': info_meta['name'][idx],
                'image_id': info_meta['id'][idx],
                'width': info_meta['width'][idx],
                'height': info_meta['height'][idx],
                'score': float(batch_cls_scores[idx][ids])
            }
            json_content.append(item)

    # print(json_content)
    # exit()
    with open(json_file, 'w') as f:
        json.dump(json_content, f)
            
    
def dataloader_inference(file_paths, idx):
    
    batch_instances = OrderedDict()
    batch_instances['meta'] = OrderedDict()
    batch_instances['meta']['id'] = []
    batch_instances['meta']['name'] = []
    batch_instances['meta']['width'] = []
    batch_instances['meta']['height'] = []
    # batch_instances['img'] = []
    img = cv2.imread(file_paths)
    batch_instances['meta']['id'].append(idx)
    batch_instances['meta']['name'].append(file_paths)
    batch_instances['meta']['width'].append(img.shape[1])
    batch_instances['meta']['height'].append(img.shape[0])
    return batch_instances


# batch_instances => dict(meta / img)
def _get_bboxes_batch(batch_instances, model):
    
    info_meta = batch_instances['meta']
    # batch_img = torch.tensor(batch_instances['img'])
    results = model.predict(source=info_meta['name'], save=False, show=False)
    
    batch_bboxes = torch.as_tensor([(result.boxes.xyxy.cpu().numpy()) for result in results])
    batch_cls_scores = torch.as_tensor([(result.boxes.conf.cpu().numpy()) for result in results])
    batch_labels = torch.as_tensor([(result.boxes.cls.cpu().numpy()) for result in results])
    batch_cls_uncertainties = -1 * (batch_cls_scores * torch.log(batch_cls_scores+1e-10) + (1-batch_cls_scores) * torch.log((1-batch_cls_scores) + 1e-10))
    batch_box_uncertainties = torch.zeros_like(batch_cls_uncertainties)
    json_file = 'uncertainty.json'
    create_json_inference(batch_bboxes, batch_cls_scores , batch_labels, batch_cls_uncertainties, info_meta, json_file)


def __preprocess__(file_paths):
    model = __init__model__(pretrain_weight = '/home/mq/data_disk2T/Thang/bak/src/runs/detect/train10/weights/best.pt')
    for idx, file_path in enumerate(file_paths):
        batch_instances = dataloader_inference(file_path, idx)
        _get_bboxes_batch(batch_instances, model)

def is_box_valid(box, img_size):
        eps = 1e-10
        size_thr=16
        ratio_thr=5
        # clip box and filter out outliers
        img_w, img_h = img_size
        x1, y1, w, h = box
        if (x1 > img_w) or (y1 > img_h):
            return False
        x2 = min(img_w, x1+w)
        y2 = min(img_h, y1+h)
        w = x2 - x1
        h = y2 - y1
        return (np.sqrt(w*h) > size_thr) and (w/(h+eps) < ratio_thr) and (h/(w+eps) < ratio_thr)

def get_class_qualities(file_path):
    # Đọc nội dung file JSON
    with open(file_path, 'r') as f:
        data = json.load(f)
    # Lấy danh sách class_quality từ file JSON
    class_qualities = data["class_quality"]

    return np.array(class_qualities)

# def _get_classwise_weight(file_path):
#     class_weight_alpha= 0.3
#     class_weight_ub= 0.2
#     class_qualities=get_class_qualities(file_path)
#     reverse_q = 1 - class_qualities
#     b = np.exp(1. / class_weight_alpha) - 1
#     _weights = 1 + class_weight_alpha * np.log(b * reverse_q + 1) * class_weight_ub

#     class_weights = dict()
#     for i in range(len(_weights)):
#             class_weights[i] = _weights[i]
#     # print( class_weights)
#     return class_weights

# def al_uncertainty_caculate(quality_path, uncertainty_path, img_path, k_samples ):
#     score_thr=0.05
#     class_weights=_get_classwise_weight(quality_path)
#     # Tạo image_uncertainties với tên tệp là key và giá trị là 0
#     image_uncertainties = {file_name: 0 for file_name in os.listdir(img_path)}
    
#     with open(uncertainty_path, 'r') as f:
#         results = json.load(f)
#     for bbox in results: 
#         img_size =(bbox["width"], bbox["height"])
#         if not is_box_valid(bbox["bbox"], img_size):
#             continue
#         if bbox['score'] < score_thr:
#             continue
#         uncertainty = float(np.sum(bbox['cls_uncertainty']))
#         label=bbox["category_id"]
#         image_uncertainties["file_name"]=image_uncertainties["file_name"]+(uncertainty*class_weights[label])
#     # Sắp xếp dictionary theo giá trị giảm dần và chọn ra n key lớn nhất
#     top_k_samples = sorted(image_uncertainties, key=image_uncertainties.get, reverse=True)[:k_samples]
#     list_k_samples = list(top_k_samples.keys())
#     return list_k_samples

file_paths = glob.glob('/home/mq/data_disk2T/Thang/bak/src/data1/val/images/*.jpg')
__preprocess__(file_paths)
# al_uncertainty_caculate("/home/mq/data_disk2T/Thang/MTagi/class_quality.json", "/home/mq/data_disk2T/Thang/MTagi/uncertainty.json",  )
