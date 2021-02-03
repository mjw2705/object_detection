import torch
import torch.nn as nn

from Vgg16 import VGG16
from Darkent53 import Darkconv, Darknet53
from utils import get_absolute_yolo_box, get_relative_yolo_box, xywh_to_x1x2y1y2, broadcast_iou


anchors_wh = torch.tensor([[10, 13], [16, 30], [33, 23],
                           [30, 61], [62, 45], [59, 119],
                           [116, 90], [156, 198], [373, 326]]).float() / 416


class YoloV3(nn.Module):
    def __init__(self, num_classes, backbone=Darknet53):
        super(YoloV3, self).__init__()
        self.num_classes = num_classes
        self.backbone = backbone

        final_filter = 3 * (4 + 1 + num_classes)

        # large_scale(y2)
        self.large_5dbl = nn.Sequential(
            Darkconv(in_c=1024, out_c=512, kernel_size=1, stride=1),
            Darkconv(in_c=512, out_c=1024, kernel_size=3, stride=1),
            Darkconv(in_c=1024, out_c=512, kernel_size=1, stride=1),
            Darkconv(in_c=512, out_c=1024, kernel_size=3, stride=1),
            Darkconv(in_c=1024, out_c=512, kernel_size=1, stride=1)
        )
        self.large_feature = nn.Sequential(
            Darkconv(in_c=512, out_c=1024, kernel_size=3, stride=1),
            nn.Conv2d(in_channels=1024, out_channels=final_filter, kernel_size=1)
        )

        # medium_scale(y1)
        self.large_upsampling = nn.Sequential(
            Darkconv(in_c=512, out_c=256, kernel_size=1, stride=1),
            nn.Upsample(scale_factor=2)
        )
        self.medium_5dbl = nn.Sequential(
            # Darkconv(in_c=512, out_c=256, kernel_size=1, stride=1),
            # concat해서 input 채널은 512 + 256
            Darkconv(in_c=768, out_c=256, kernel_size=1, stride=1),
            Darkconv(in_c=256, out_c=512, kernel_size=3, stride=1),
            Darkconv(in_c=512, out_c=256, kernel_size=1, stride=1),
            Darkconv(in_c=256, out_c=512, kernel_size=3, stride=1),
            Darkconv(in_c=512, out_c=256, kernel_size=1, stride=1)
        )
        self.medium_feature = nn.Sequential(
            Darkconv(in_c=256, out_c=512, kernel_size=3, stride=1),
            nn.Conv2d(in_channels=512, out_channels=final_filter, kernel_size=1)
        )

        # small_scale(y0)
        self.medium_upsampling = nn.Sequential(
            Darkconv(in_c=256, out_c=128, kernel_size=1, stride=1),
            nn.Upsample(scale_factor=2)
        )
        self.small_5dbl = nn.Sequential(
            # Darkconv(in_c=256, out_c=128, kernel_size=1, stride=1),
            Darkconv(in_c=384, out_c=128, kernel_size=1, stride=1),
            Darkconv(in_c=128, out_c=256, kernel_size=3, stride=1),
            Darkconv(in_c=256, out_c=128, kernel_size=1, stride=1),
            Darkconv(in_c=128, out_c=256, kernel_size=3, stride=1),
            Darkconv(in_c=256, out_c=128, kernel_size=1, stride=1)
        )
        self.small_feature = nn.Sequential(
            Darkconv(in_c=128, out_c=256, kernel_size=3, stride=1),
            nn.Conv2d(in_channels=256, out_channels=final_filter, kernel_size=1)
        )

    def forward(self, inputs):
        # y0, y1, y2
        x_s, x_m, x_l = self.backbone

        x = self.large_5dbl(x_l)
        y_large = self.large_feature(x)

        x = self.large_upsampling(x)
        x = torch.cat((x, x_m), 1)
        x = self.medium_5dbl(x)
        y_medium = self.medium_feature(x)

        x = self.medium_upsampling(x)
        x = torch.cat((x, x_s), 1)
        x = self.small_5dbl(x)
        y_small = self.small_feature(x)

        # (batch, 3(anchor box) x (5 + num_classes), grid, grid) 형태
        y_small_shape = y_small.shape
        y_medium_shape = y_medium.shape
        y_large_shape = y_large.shape

        # (batch, grid, grid, 3, 5 + num_classes)로 변환환
        y_small = y_small.view(y_small_shape[0], -1, 3, y_small_shape[-2], y_small_shape[-1]).permute(0, -1, -2, 2, 1)
        y_medium = y_medium.view(y_medium_shape[0], -1, 3, y_medium_shape[-2], y_medium_shape[-1]).permute(0, -1, -2, 2, 1)
        y_large = y_large.view(y_large_shape[0], -1, 3, y_large_shape[-2], y_large_shape[-1]).permute(0, -1, -2, 2, 1)

        # if training:
        #     return y_small, y_medium, y_large
        #
        box_small = get_absolute_yolo_box(y_small, anchors_wh[0:3], self.num_classes)
        box_medium = get_absolute_yolo_box(y_medium, anchors_wh[3:6], self.num_classes)
        box_large = get_absolute_yolo_box(y_large, anchors_wh[6:9], self.num_classes)

        return box_small, box_medium, box_large


class Yololoss(nn.Module):
    def __init__(self, num_classes, valid_anchors_wh, ignore_thresh=0.5, lambda_coord=5.0, lambda_noobj=0.5):
        super(Yololoss, self).__init__()
        self.num_classes = num_classes
        self.valid_anchors_wh = valid_anchors_wh
        self.ignore_thresh = ignore_thresh
        self.lambda_coord = lambda_coord
        self.lambda_noobj = lambda_noobj

    def forward(self, y_true, y_pred):
        # iou, ignore_mask 계산에 필요
        pred_box_abs, pred_obj, pred_class, pred_box_rel = get_absolute_yolo_box(y_pred,
                                                                                 self.valid_anchors_wh,
                                                                                 self.num_classes)
        pred_box_abs = xywh_to_x1x2y1y2(pred_box_abs)
        pred_xy_rel = torch.sigmoid(pred_box_rel[..., 0:2])
        pred_wh_rel = pred_box_rel[..., 2:4]

        # loss 계산에 필요
        true_box_rel, true_obj, true_class, true_box_abs = get_relative_yolo_box(y_true,
                                                                                 self.valid_anchors_wh,
                                                                                 self.num_classes)
        true_box_abs = xywh_to_x1x2y1y2(true_box_abs)
        true_xy_rel = true_box_rel[..., 0:2]
        true_wh_rel = true_box_rel[..., 2:4]

        true_wh_abs = true_box_abs[..., 2:4]

        # w, h를 통해 작은 box detect를 위한 조정
        weight = 2 - true_wh_abs[..., 0] * true_wh_abs[..., 1]

        xy_loss = self.calc_xywh_loss(true_xy_rel, pred_xy_rel, true_obj, weight)
        wh_loss = self.calc_xywh_loss(true_wh_rel, pred_wh_rel, true_obj, weight)
        ignore_mask = self.calc_ignore_mask(true_obj, true_box_abs, pred_box_abs)
        obj_loss = self.calc_obj_loss(true_obj, pred_obj, ignore_mask)
        class_loss = self.calc_class_loss(true_obj, true_class, pred_class)

        return xy_loss + wh_loss + class_loss + obj_loss, (xy_loss, wh_loss, class_loss, obj_loss)

    def calc_xywh_loss(self, true, pred, true_obj, weight):
        loss = torch.sum(torch.square(true, pred), dim=-1)
        true_obj = torch.squeeze(true_obj, dim=-1)
        loss = loss * true_obj * weight
        loss = torch.sum(loss, dim=(1, 2, 3)) * self.lambda_coord

        return loss

    def calc_ignore_mask(self, true_obj, true_box, pred_box):
        # true_box, pred_box는 [xmin, ymin, xmax, ymax]
        obj_mask = torch.squeeze(true_obj, dim=-1)

        best_iou = []
        for x in zip(pred_box, true_box, obj_mask):
            # obj_mask가 true(1)인 true_box만 mask에 넣는다.
            mask = x[1][x[2].bool()]
            if mask.size(0) is not 0:
                best_iou.append(broadcast_iou(x[0], mask))
            else:
                best_iou.append(torch.zeros(true_box.shape[1:4]).cuda())
        best_iou = torch.stack(best_iou)

        ignore_mask = (best_iou < self.ignore_thresh).float()
        ignore_mask = ignore_mask.unsqueeze(-1)

        # ignore_mask = 0이면 무시 / 1이면 안무시
        return ignore_mask

    def calc_obj_loss(self, true_obj, pred_obj, ignore_mask):
        obj_entropy = self.binary_cross_entropy(pred_obj, true_obj)
        obj_loss = obj_entropy * true_obj
        noobj_loss = (1 - true_obj) * obj_entropy * ignore_mask

        obj_loss = torch.sum(obj_loss, dim=(1, 2, 3, 4))
        noobj_loss = torch.sum(noobj_loss, dim=(1, 2, 3, 4)) * self.lambda_noobj

        return obj_loss + noobj_loss

    def calc_class_loss(self, true_obj, true_class, pred_class):
        class_loss = self.binary_cross_entropy(pred_class, true_class)
        class_loss = class_loss * true_obj
        class_loss = torch.sum(class_loss, dim=(1, 2, 3, 4))

        return class_loss

    def binary_cross_entropy(self, logits, labels):
        epsilon = 1e-7
        logits = torch.clamp(logits, epsilon, 1 - epsilon)

        return -(labels * torch.log(logits) + (1 - labels) * torch.log(logits))


def main():
    inputs = torch.randn(1, 3, 416, 416)

    model = YoloV3(20)
    outputs = model(inputs)
    y_small, y_medium, y_large = outputs

    print(y_small[0].shape)
    print(y_small[1].shape)
    print(y_small[2].shape)
    print(y_small[3].shape)

    # loss = Yololoss(10, anchors_wh[0])
    # y_pred = model(inputs)
    # loss1, loss_breakdonw = loss(label, y_pred)


if __name__ == '__main__':
    main()


