# data를 입력에 맞게 변환
# 네트워크의 output은 grid x grid x 3 x (5+num_classes) 3개
# gt와 예측 사이의 델타를 계산하기 위해 gt를 매트릭스로
# gt bbox에 대한 best scale anchor를 선택해야함
# gt bbox와 모든 앵커가 같은 중심을 공유한다고 가정할 때, 잘 일치하는 정도는 최소 너비 x 최소 높이인 겹치는 영역
import os
import cv2
import numpy as np
import torch
import xml.etree.ElementTree as ET
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


anchors_wh = torch.tensor([[10, 13], [16, 30], [33, 23],
                           [30, 61], [62, 45], [59, 119],
                           [116, 90], [156, 198], [373, 326]]).float() / 416

DB_path = 'C:/Users/mjw27/Desktop/object_dectection/data/ex'
classes = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]


class CustomDataset(Dataset):
    def __init__(self, num_classes, file_dir, output_shape=(416, 416)):
        super(CustomDataset, self).__init__()
        # self.img_dir = img_dir
        self.num_classes = num_classes
        self.output_shape = output_shape

        files = os.listdir(file_dir)
        if 'JPEGImages' in files:
            self.img_dir = os.path.join(file_dir, 'JPEGImages')
        else:
            raise Exception('No exist the \'jpegimages\' file')

        if 'Annotations' in files:
            anno_dir = os.path.join(file_dir, 'Annotations')
        else:
            raise Exception('No exist the \'annotation\' file')

        self.all_annos = self.parse_annotation(anno_dir)

    def __len__(self):
        return len(self.all_annos)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = os.path.join(self.img_dir, self.all_annos[idx]['filename'])
        image = cv2.resize(cv2.imread(img_path), self.output_shape) / 255
        image = torch.from_numpy(image).permute(2, 0, 1).float()

        classes, bboxes = self.parse_y_feature(self.all_annos[idx])

        label = (self.preprocess_label_for_one_class(bboxes, classes, 52, torch.tensor([0, 1, 2])),
                 self.preprocess_label_for_one_class(bboxes, classes, 26, torch.tensor([3, 4, 5])),
                 self.preprocess_label_for_one_class(bboxes, classes, 13, torch.tensor([6, 7, 8])))

        return image, label

    # class, bbox의 주석에 하나의 스케일에 대해 원하는 형식으로 전처리(grid, anchor, x, y, w, h, obj, one-hot class.. etc)
    def preprocess_label_for_one_class(self, bbox, classes, grid_size, valid_anchor):
        y = torch.zeros((grid_size, grid_size, 3, 5 + self.num_classes))
        anchor_idx = self.find_best_anchor(bbox)

        num_boxes = classes.shape[0]
        for i in range(num_boxes):
            curr_class = torch.tensor(classes[i]).float()
            curr_box = torch.tensor(bbox[i]).float()
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
                index = torch.tensor([grid_cell_xy[1], grid_cell_xy[0], adjusted_anchor_idx]).int()
                update = torch.cat((curr_box_xy, curr_box_wh, torch.tensor([1.0]).float(), curr_class), dim=0)

                # gird cell 하나에 대한 anchor 3개
                y[index[0]][index[1]][index[2]] = update

        return y

    # input은 (num_boxes, 4)인 ground truth box
    # gt box와 anchor box의 iou가 가장 큰 anchor 구하기
    def find_best_anchor(self, bbox):
        print(bbox)
        box_wh = torch.from_numpy(bbox[..., 2:4] - bbox[..., 0:2]).float()
        box_wh = box_wh.unsqueeze(1).repeat(1, anchors_wh.shape[0], 1).float()  # anchors_wh.shape[0] = 9

        intersection = torch.min(box_wh[..., 0], anchors_wh[..., 0]) * torch.min(box_wh[..., 1], anchors_wh[..., 1])
        box_area = box_wh[..., 0] * box_wh[..., 1]
        anchor_area = anchors_wh[..., 0] * anchors_wh[..., 1]

        iou = intersection / (box_area + anchor_area - intersection)
        anchor_idx = torch.argmax(iou, dim=-1).int()

        return anchor_idx

    def parse_y_feature(self, annos):
        bboxes = []
        classes = []
        labels = torch.arange(self.num_classes).view(self.num_classes, 1)

        for obj in annos['object']:
            bboxes.append([obj['xmin'], obj['ymin'], obj['xmax'], obj['ymax']])

        return np.array(classes), np.array(bboxes)

    def parse_annotation(self, anno_dir):
        all_anns = []

        for ann in sorted(os.listdir(anno_dir)):
            tree = ET.parse(os.path.join(anno_dir, ann))
            img = {'object': []}

            for elem in tree.iter():
                if 'filename' in elem.tag:
                    img['filename'] = elem.text
                if 'width' in elem.tag:
                    img['width'] = int(elem.text)
                if 'height' in elem.tag:
                    img['height'] = int(elem.text)
                if 'depth' in elem.tag:
                    img['depth'] = int(elem.text)
                if 'object' in elem.tag:
                    obj = {}

                    for att in list(elem):
                        if 'name' in att.tag:
                            img['object'] += [obj]
                            obj['name'] = att.text
                            obj['cls_id'] = classes.index(obj['name'])
                        if 'bndbox' in att.tag:
                            for dim in list(att):
                                if 'xmin' in dim.tag:
                                    obj['xmin'] = int(dim.text)
                                if 'ymin' in dim.tag:
                                    obj['ymin'] = int(dim.text)
                                if 'xmax' in dim.tag:
                                    obj['xmax'] = int(dim.text)
                                if 'ymax' in dim.tag:
                                    obj['ymax'] = int(dim.text)

            if len(img['object']) > 0:
                all_anns.append(img)

        return all_anns


if __name__=='__main__':
    dataset = CustomDataset(20, DB_path)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    image, label = dataset

