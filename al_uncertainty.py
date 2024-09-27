import os
import cv2
import glob
import json
import torch 
from ultralytics import YOLO
from collections import OrderedDict
from al_quality import load_bboxes_pred

def __init__model__(pretrain_weight = ''):
    
    model = YOLO(pretrain_weight)
    return model

# coco format save 
# images
# super_category => no need
# annotation


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
                'cls_uncertainty': batch_cls_uncertainty[idx].tolist(),
                'file_name': info_meta['name'][idx],
                'image_id': info_meta['id'][idx]
            }
            json_content.append(item)

    # print(json_content)
    # exit()
    with open(json_file, 'w') as f:
        json.dump(json_content, f)
            
    
def dataloader_inference(file_paths):
    
    batch_instances = OrderedDict()
    batch_instances['meta'] = OrderedDict()
    batch_instances['meta']['id'] = []
    batch_instances['meta']['name'] = []
    # batch_instances['img'] = []
    for idx, file_path in enumerate(file_paths):
        batch_instances['meta']['id'].append(idx)
        batch_instances['meta']['name'].append(file_path)
    return batch_instances


# batch_instances => dict(meta / img)
def _get_bboxes_batch(batch_instances, model):
    
    info_meta = batch_instances['meta']
    # batch_img = torch.tensor(batch_instances['img'])
    results = model.predict(source=info_meta['name'], save=False, show=True)
    batch_bboxes = torch.as_tensor([(result.boxes.xyxy.cpu().numpy()) for result in results])
    batch_cls_scores = torch.as_tensor([(result.boxes.conf.cpu().numpy()) for result in results])
    batch_labels = torch.as_tensor([(result.boxes.cls.cpu().numpy()) for result in results])
    batch_cls_uncertainties = -1 * (batch_cls_scores * torch.log(batch_cls_scores+1e-10) + (1-batch_cls_scores) * torch.log((1-batch_cls_scores) + 1e-10))
    batch_box_uncertainties = torch.zeros_like(batch_cls_uncertainties)
    json_file = 'uncertainty.json'
    create_json_inference(batch_bboxes, batch_cls_scores , batch_labels, batch_cls_uncertainties, info_meta, json_file)


def __main__(file_paths):
    
    model = __init__model__(pretrain_weight = '/home/mq/data_disk2T/Thang/runs/detect/train3/weights/best.pt')
    batch_instances = dataloader_inference(file_paths)
    _get_bboxes_batch(batch_instances, model)


file_paths = glob.glob('/home/mq/data_disk2T/Thang/runs/detect/predict/*.jpg')
__main__(file_paths)