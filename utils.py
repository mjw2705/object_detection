import torch
import numpy as np


# 예측한 상대좌표를 절대좌표로
# input인 y_pred는 (batch, grid, grid, 3, 5 + num_classes) 형태
def get_absolute_yolo_box(y_pred, valid_anchors_wh, num_classes):
    t_xy, t_wh, objectness, classes = torch.split(y_pred, (2, 2, 1, num_classes), dim=-1)

    b_xy = torch.sigmoid(t_xy)
    b_wh = t_wh
    objectness = torch.sigmoid(objectness)
    classes = torch.sigmoid(classes)
    bbox_rel = torch.cat((b_xy, b_wh), dim=-1)

    grid_size = y_pred.shape[1]
    grid_x, grid_y = torch.meshgrid(torch.arange(end=grid_size, dtype=torch.float, device=b_xy.device),
                                    torch.arange(end=grid_size, dtype=torch.float, device=b_xy.device))
    C_xy = torch.stack((grid_y, grid_x), dim=-1).unsqueeze_(2)

    b_xy = b_xy + C_xy
    b_xy = b_xy / float(grid_size)  # 정규화
    b_wh = torch.exp(b_wh) * valid_anchors_wh

    bbox_abs = torch.cat((b_xy.float(), b_wh.float()), dim=-1)

    return bbox_abs, objectness, classes, bbox_rel


# 절대좌표를 상대좌표로
def get_relative_yolo_box(y_true, valid_anchors_wh, num_classes):
    b_xy, b_wh, objectness, classes = torch.split(y_true, (2, 2, 1, num_classes), dim=-1)
    bbox_abs = torch.cat((b_xy, b_wh), dim=-1)

    grid_size = y_true.shape[1]
    grid_x, grid_y = torch.meshgrid(torch.arange(end=grid_size, dtype=torch.float, device=b_xy.device),
                                    torch.arange(end=grid_size, dtype=torch.float, device=b_xy.device))
    C_xy = torch.stack((grid_y, grid_x), dim=-1).unsqueeze_(2)

    b_xy = y_true[..., 0:2]
    b_wh = y_true[..., 2:4]

    t_xy = b_xy * float(grid_size) - C_xy
    t_wh = torch.log(b_wh / valid_anchors_wh)
    t_wh = torch.where(torch.logical_or(torch.isinf(t_wh), torch.isnan(t_wh)),
                       torch.zeros_like(t_wh), t_wh)
    # isinf(무한인지 아닌지) / logical_or이 true이면 0으로 채워서, 아니면 t_wh

    bbox_rel = torch.cat((t_xy, t_wh), dim=-1)

    return bbox_rel, objectness, classes, bbox_abs


def xywh_to_x1x2y1y2(box):
    xy = box[..., 0:2]
    wh = box[..., 2:4]

    x1y1 = xy - wh / 2
    x2y2 = xy + wh / 2

    y_box = torch.cat((x1y1, x2y2), dim=-1)

    return y_box


# broadcast_iou(pred_box, true_box) / pred_box, trud_box의 마지막 차원은 [xmin, ymin, xmax, ymax]
def broadcast_iou(box1, box2):
    box_1, box_2 = torch.broadcast_tensors(box1, box2)

    ze = torch.zeros(box_1[..., 2].shape).cuda()
    # 두 박스가 안겹쳐 있으면 0
    int_w = torch.max(torch.min(box_1[..., 2], box_2[..., 2])
                      - torch.max(box_1[..., 0], box_2[..., 0]), ze)
    int_h = torch.max(torch.min(box_1[..., 3], box_2[..., 3])
                      - torch.max(box_1[..., 1], box_2[..., 1]), ze)

    intersec_area = int_w * int_h

    box_1_area = (box_1[..., 2] - box_1[..., 0]) * \
                 (box_1[..., 3] - box_1[..., 1])
    box_2_area = (box_2[..., 2] - box_2[..., 0]) * \
                 (box_2[..., 3] - box_2[..., 1])

    iou = intersec_area / (box_1_area + box_2_area - intersec_area)

    return iou


def broadcast_ioutf(box1, box2):
    # (batch, 507, 1, 4)
    box_1 = torch.unsqueeze(box1, dim=-2)
    # (batch, 1, 507, 4)
    box_2 = torch.unsqueeze(box2, dim=-3)
    # (batch, 507, 507, 4)
    boxa, boxb = torch.broadcast_tensors(box_1, box_2)

    al, at, ar, ab = torch.chunk(boxa, 4, dim=-1)
    bl, bt, br, bb = torch.chunk(boxb, 4, dim=-1)

    left = torch.maximum(al, bl)
    right = torch.maximum(ar, br)
    top = torch.maximum(at, bt)
    bottom = torch.maximum(ab, bb)

    iw = torch.clamp(right - left, min=0, max=1)
    ih = torch.clamp(bottom - top, min=0, max=1)

    intersect = iw * ih
    area_a = (ar - al) * (ab - at)
    area_b = (br - bl) * (bb - bt)
    union = area_a + area_b - intersect

    iou = torch.squeeze(intersect / union + 1e-7, dim=-1)

    return iou


