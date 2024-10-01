import os
import json
import torch
import numpy as np
from quality import Class_Quality
from collections import OrderedDict

# rewrite in config file
CLASSES=[]
n_images = 200
score_thr = 0.5

def _get_classwise_weight(self, results_json):
        ckpt_path = os.path.join(
           os.path.dirname(results_json), 'latest.pth'
        )
        ckpt = torch.load(ckpt_path, map_location='cpu')
        # replace this by Yolo head.class_quality
        class_qualities = ckpt['state_dict']['bbox_head.class_quality'].numpy()
        reverse_q = 1 - class_qualities
        b = np.exp(1. / self.class_weight_alpha) - 1
        _weights = 1 + self.class_weight_alpha * np.log(b * reverse_q + 1) * self.class_weight_ub

        class_weights = dict()
        for i in range(len(_weights)):
            cid = self.class_name2id[self.CLASSES[i]]
            class_weights[cid] = _weights[i]
        return class_weights
    

def al_acquisition(self, result_json, last_label_path):

        # class_weights = self._get_classwise_weight(result_json)
        # class_weights['0']

        with open(result_json) as f:
            results = json.load(f)

        category_uncertainty = OrderedDict()
        category_count = OrderedDict()
        image_uncertainties = OrderedDict()
        for res in results:
            img_id = res['image_id']
            image_uncertainties[img_id] = [0.]
            img_size = (res['width'], res['height'])
            # if not self.is_box_valid(res['bbox'],img_size):
            #     continue
            if res['score'] < score_thr:
                continue
            uncertainty = float(res['cls_uncertainty'])
            label = res['category_id']
            if label not in category_uncertainty.keys():
                category_uncertainty[label] = 0.
                category_count[label] = 0.
            category_uncertainty[label] += uncertainty
            category_count[label] += 1

        category_avg_uncertainty = OrderedDict()
        for k in category_uncertainty.keys():
            category_avg_uncertainty[k] = category_uncertainty[k] / (category_count[k] + 1e-5)

        # with open(last_label_path) as f:
        #     last_labeled_data = json.load(f)
        #     last_labeled_img_ids = [x['id'] for x in last_labeled_data['images']]

        
        # for img_id in res['image_id']:
        #     image_hit[img_id] = 0
        # for img_id in last_labeled_img_ids:
        #     image_hit[img_id] = 1

        # image_uncertainties = OrderedDict()
        # for img_id in self.oracle_data.keys():
        #     if image_hit[img_id] == 0:
        #         image_uncertainties[img_id] = [0.]

        for res in results:
            img_id = res['image_id']
            img_size = (res['width'], res['height'])
            # if not self.is_box_valid(res['bbox'], img_size):
            #     continue
            if res['score'] < score_thr:
                continue
            uncertainty = float(res['cls_uncertainty'])
            label = res['category_id']
            image_uncertainties[img_id].append(uncertainty * class_weights[label])

        for img_id in image_uncertainties.keys():
            _img_uncertainties = np.array(image_uncertainties[img_id])
            image_uncertainties[img_id] = _img_uncertainties.sum()

        img_ids = []
        merged_img_uncertainties = []
        for k, v in image_uncertainties.items():
            img_ids.append(k)
            merged_img_uncertainties.append(v)
        img_ids = np.array(img_ids)
        merged_img_uncertainties = np.array(merged_img_uncertainties)

        inds_sort = np.argsort(-1. * merged_img_uncertainties)
        sampled_inds = inds_sort[:n_images]
        unsampled_img_ids = inds_sort[n_images:]
        sampled_img_ids = img_ids[sampled_inds].tolist()
        unsampled_img_ids = img_ids[unsampled_img_ids].tolist()

        return sampled_img_ids, unsampled_img_ids
