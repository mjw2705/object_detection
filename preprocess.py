# data를 입력에 맞게 변환
# 네트워크의 output은 grid x grid x 3 x (5+num_classes) 3개
# gt와 예측 사이의 델타를 계산하기 위해 gt를 매트릭스로
# gt bbox에 대한 best scale anchor를 선택해야함
# gt bbox와 모든 앵커가 같은 중심을 공유한다고 가정할 때, 잘 일치하는 정도는 최소 너비 x 최소 높이인 겹치는 영역
import os
import csv
import cv2
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
import xml.etree.ElementTree as ET
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


anchors_wh = torch.tensor([[10, 13], [16, 30], [33, 23],
                           [30, 61], [62, 45], [59, 119],
                           [116, 90], [156, 198], [373, 326]]).float().cuda() / 416

# DB_path = './data/VOC2007_trainval'
# csv_file = '2007_train.csv'
DB_path = './data/ex'
csv_file = 'ex_train.csv'


class CustomDataset(Dataset):
    def __init__(self, DB_path, csv_file, num_classes, output_shape=(416, 416)):
        self.label_csv = pd.read_csv(f'{csv_file}')
        self.num_classes = num_classes
        self.output_shape = output_shape
        self.DB_path = DB_path

    def __len__(self):
        return len(self.label_csv)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        try:
            img_path = os.path.join(DB_path, self.label_csv.iloc[idx, 0])
            image = cv2.resize(cv2.imread(img_path), self.output_shape) / 255
            image = torch.from_numpy(image).permute(2, 0, 1).float()

        except Exception as e:
            print("Invalid")

        bboxes, classes = self.parse_y_features(idx)

        label = (
            self.preprocess_label_for_one_class(bboxes, classes, 52, torch.tensor([0, 1, 2]).cuda()),
            self.preprocess_label_for_one_class(bboxes, classes, 26, torch.tensor([3, 4, 5]).cuda()),
            self.preprocess_label_for_one_class(bboxes, classes, 13, torch.tensor([6, 7, 8]).cuda())
        )

        return image, label

    def parse_y_features(self, idx):
        sizes = self.label_csv.iloc[idx, 1]
        sizes = list(map(int, sizes.split(' ')))

        bboxs = self.label_csv.iloc[idx, 2]
        bboxs = list(map(str, bboxs.split(',')))
        bbox = []
        for bx in bboxs:
            normal = list(map(int, bx.split(' ')))
            xmin = normal[0] / sizes[0]
            ymin = normal[1] / sizes[1]
            xmax = normal[2] / sizes[0]
            ymax = normal[3] / sizes[1]
            bbox.append([xmin, ymin, xmax, ymax])
        bbox = np.array(bbox)

        labels = torch.arange(self.num_classes)
        one_hot = torch.nn.functional.one_hot(labels)
        clss = self.label_csv.iloc[idx, 3]
        clss = str(clss)
        clss = list(map(int, clss.split(',')))
        cls = one_hot[clss]
        cls = np.array(cls)

        return bbox, cls

    # class, bbox의 주석에 하나의 스케일에 대해 원하는 형식으로 전처리(grid, anchor, x, y, w, h, obj, one-hot class.. etc)
    def preprocess_label_for_one_class(self, bbox, classes, grid_size, valid_anchor):
        y = torch.zeros((grid_size, grid_size, 3, 5 + self.num_classes)).cuda()
        anchor_idx = self.find_best_anchor(bbox)

        num_boxes = classes.shape[0]
        for i in range(num_boxes):
            curr_class = torch.tensor(classes[i]).cuda().float()
            curr_box = torch.tensor(bbox[i]).cuda().float()
            curr_anchor = anchor_idx[i]

            # 앵커는 현재 그리드 사이즈에 맞는 것만 사용, 인덱스 3개로 조정
            if curr_anchor in valid_anchor:
                adjusted_anchor_idx = curr_anchor % 3

                # loss 계산을 위해 (xmin, ymin, xmax, ymax)를 (x, y, w, h)로 변환
                curr_box_xy = (curr_box[..., 0:2] + curr_box[..., 2:4]) / 2
                curr_box_wh = curr_box[..., 2:4] - curr_box[..., 0:2]

                # grid cell의 index
                grid_cell_xy = curr_box_xy // float(1 / grid_size)
                # gird[y][x][anchor] 형태 = (tx, ty, bw, bh, obj, class)
                index = torch.tensor([grid_cell_xy[1], grid_cell_xy[0], adjusted_anchor_idx]).int().cuda()
                update = torch.cat((curr_box_xy, curr_box_wh, torch.tensor([1.0]).cuda().float(), curr_class), dim=0)

                # gird cell 하나에 대한 anchor 3개
                y[index[0]][index[1]][index[2]] = update

        return y

    # input은 (num_boxes, 4)인 ground truth box
    # gt box와 anchor box의 iou가 가장 큰 anchor 구하기
    def find_best_anchor(self, bbox):
        box_wh = torch.from_numpy(bbox[..., 2:4] - bbox[..., 0:2]).float().cuda()
        box_wh = box_wh.unsqueeze(1).repeat(1, anchors_wh.shape[0], 1).float()  # anchors_wh.shape[0] = 9

        intersection = torch.min(box_wh[..., 0], anchors_wh[..., 0]) * torch.min(box_wh[..., 1], anchors_wh[..., 1])
        box_area = box_wh[..., 0] * box_wh[..., 1]
        anchor_area = anchors_wh[..., 0] * anchors_wh[..., 1]

        iou = intersection / (box_area + anchor_area - intersection)
        anchor_idx = torch.argmax(iou, dim=-1).int()

        return anchor_idx


if __name__=='__main__':
    dataset = CustomDataset(DB_path, csv_file, 20)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    for batch, data in enumerate(dataloader):
        image, label = data

        print(image.shape, label[0].shape, label[1].shape, label[2].shape)







