import os
import datetime
import torch
from torch.utils.data import DataLoader
from utils import get_absolute_yolo_box

from Yolov3 import YoloV3, Yololoss
from Yolov3 import anchors_wh, anchors_wh_mask
from preprocess import CustomDataset


BATCH_SIZE = 1
EPOCH = 1000

classes = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]


def main():
    # DB_path = './data/VOC2007_trainval/'
    # csv_file = '2007_train.csv'
    DB_path = './data/ex'
    csv_file = 'ex_train.csv'

    pth_dir = './models/'

    # # 학습했던 모델 불러오기
    saved_pth_dir = './models'
    pth_file = '999_0.8255.pth'

    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda' if torch.cuda.is_available() else torch.device('cpu'))

    num_classes = 20
    lr_rate = 0.001

    dataset = CustomDataset(DB_path, csv_file, num_classes)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

    model = YoloV3(num_classes).to(device)
    # loss_object = [Yololoss(num_classes, valid_anchors_wh) for valid_anchors_wh in anchors_wh_mask]
    optimizer = torch.optim.Adam(model.parameters(), lr=lr_rate)
    loss1 = Yololoss(num_classes, anchors_wh_mask[2]).cuda()

    # EPOCH = 1

    true_idxs = [[9, 4, 2], [9, 10, 1], [8, 1, 1], [4, 2, 1], [6, 4, 2], [6, 10, 1]]
    for epoch in range(EPOCH):
        model.train()
        for image, label in dataloader:
            optimizer.zero_grad()
            if use_cuda:
                image = image.cuda()
                # label[0] = label[0].cuda()
                # label[1] = label[1].cuda()
                label[2] = label[2].cuda()

            outputs = model(image, training=True)
            y_small, y_medium, y_large = outputs

            total_loss, each_loss = loss1(label[2], y_large)
            total_loss.backward()
            optimizer.step()

            print(f'Epoch : {epoch} - loss : {total_loss}')
            print(f'obj_loss : {each_loss[3]}')

            bbox_abs, objectness, classes, bbox_rel = get_absolute_yolo_box(y_large, anchors_wh[6:9], num_classes)

            for true_idx in true_idxs[:1]:
                print(true_idx)
                print('Label : ', label[2][0][true_idx[0]][true_idx[1]][true_idx[2]][5:])
                print('Predic: ', y_large[0][true_idx[0]][true_idx[1]][true_idx[2]][5:])
                # print(f'Predict : {bbox_abs[0][9][4][2]} - {objectness[0][9][4][2]}')
            print()

def train_one_epoch(model, loss_object, dataloader, optimizer, use_cuda):
    len_dataloader = len(dataloader)
    epoch_total_loss = 0.0
    epoch_xy_loss = 0.0
    epoch_wh_loss = 0.0
    epoch_obj_loss = 0.0
    epoch_class_loss = 0.0

    for i, data in enumerate(dataloader):
        image, label = data

        if use_cuda:
            image = image.cuda()
            label[0] = label[0].cuda()
            label[1] = label[1].cuda()
            label[2] = label[2].cuda()
        model_output = model(image, training=True)

        total_losses, xy_losses, wh_losses, class_losses, obj_losses = [], [], [], [], []

        for loss_obj, y_pred, y_true in zip(loss_object, model_output, label):
            total_loss, each_loss = loss_obj(y_true, y_pred)
            xy_loss, wh_loss, class_loss, obj_loss = each_loss
            total_losses.append(total_loss * (1. / BATCH_SIZE))
            xy_losses.append(xy_loss * (1. / BATCH_SIZE))
            wh_losses.append(wh_loss * (1. / BATCH_SIZE))
            class_losses.append(class_loss * (1. / BATCH_SIZE))
            obj_losses.append(obj_loss * (1. / BATCH_SIZE))

        total_loss = torch.sum(torch.stack(total_losses))
        total_xy_loss = torch.sum(torch.stack(xy_losses))
        total_wh_loss = torch.sum(torch.stack(wh_losses))
        total_class_loss = torch.sum(torch.stack(class_losses))
        total_obj_loss = torch.sum(torch.stack(obj_losses))

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        epoch_total_loss += ((total_loss.item()) / len_dataloader)
        epoch_xy_loss += ((total_xy_loss.item()) / len_dataloader)
        epoch_wh_loss += ((total_wh_loss.item()) / len_dataloader)
        epoch_class_loss += ((total_class_loss.item()) / len_dataloader)
        epoch_obj_loss += ((total_obj_loss.item()) / len_dataloader)

    print(f'total_loss: {epoch_total_loss:.4f},  xy: {epoch_xy_loss:.4f}, wh: {epoch_wh_loss:.4f}, class: {epoch_class_loss:.4f}, obj: {epoch_obj_loss:.4f}')

    return epoch_total_loss


if __name__ == '__main__':
    main()
