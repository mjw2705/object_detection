import os

import cv2
import torch
from torch.utils.data import DataLoader

from Yolov3 import YoloV3, Yololoss
from Yolov3 import anchors_wh, anchors_wh_mask
from preprocess import CustomDataset
from postprocess import Postprocess


classes_name = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]

saved_pth_dir = './models'
pth_file = 'yolov3_voc2007_darknet.pth'

num_classes = 20
BATCH_SIZE = 1

img = cv2.imread('./data/VOC2007_trainval/JPEGImages/000017.jpg')
img = cv2.resize(img, (416, 416))
x = torch.from_numpy(img / 255).float().unsqueeze(0).permute(0, -1, 1, 2).cuda()

model = YoloV3(num_classes)
state = torch.load(os.path.join(saved_pth_dir, pth_file))

model.load_state_dict(state['model_state_dict'])
model.eval()
model.cuda()

y_pred = model(x, training=False)

postprocessor = Postprocess(iou_threshold=0.5, score_threshold=0.5, max_detection=10).cuda()
boxes, scores, classes, num_detection = postprocessor(y_pred)

num_img = num_detection.shape[0]

for n_img in range(num_img):
    h, w, c = img.shape
    score = scores.cpu()[n_img]

    classs = classes.cpu()[n_img]
    classs = torch.argmax(classs, dim=1)

    for i in range(num_detection.item()):
        box = boxes.cpu()[n_img][i]
        score_value = score[i].item()
        class_value = classes_name[classs[i]]
        print(score_value, class_value)

        xmin = int(box[0] * w)
        ymin = int(box[1] * h)
        xmax = int(box[2] * w)
        ymax = int(box[3] * h)

        img = cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 0, 255), 1)
        cv2.putText(img, class_value, (xmin+3, ymin+10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

    cv2.imshow('t', img)
    cv2.waitKey()
    cv2.destroyAllWindows()