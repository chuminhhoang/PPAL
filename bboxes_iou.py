import torch 
def fp16_clamp(x, min=None, max=None):
    if x.dtype == torch.float16:
        # clamp for cpu float16, tensor fp16 has no clamp implementation
        return x.float().clamp(min, max).half()
    return x.clamp(min, max)
def bbox_overlaps(bboxes1, bboxes2,eps=1e-6):
    """Calculate overlap between two set of bboxes."""
    
    # Either the boxes are empty or the length of boxes' last dimension is 4
    if (bboxes1.size(-1) == 4):
        return torch.tensor([]), False
    if (bboxes2.size(-1) == 4):
        return torch.tensor([]), False
    rows = bboxes1.size(0)
    cols = bboxes2.size(0)

    if rows * cols == 0:
        return torch.zeros(rows) 

    area1 = (bboxes1[..., 2] - bboxes1[..., 0]) * (
        bboxes1[..., 3] - bboxes1[..., 1])
    area2 = (bboxes2[..., 2] - bboxes2[..., 0]) * (
        bboxes2[..., 3] - bboxes2[..., 1])

    lt = torch.max(bboxes1[..., :2], bboxes2[..., :2])  # [B, rows, 2]
    rb = torch.min(bboxes1[..., 2:], bboxes2[..., 2:])  # [B, rows, 2]

    wh = fp16_clamp(rb - lt, min=0)
    overlap = wh[..., 0] * wh[..., 1]

    union = area1 + area2 - overlap
    
    eps = union.new_tensor([eps])
    union = torch.max(union, eps)
    ious = overlap / union
    return ious, True
   