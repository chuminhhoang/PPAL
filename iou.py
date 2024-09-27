from calculator import bbox_iou
def iou(pred_bboxes,  target_bboxes, fg_mask):
        """IoU loss."""
        iou = bbox_iou(pred_bboxes[fg_mask], target_bboxes[fg_mask], xywh=False, CIoU=True)
        return iou
