import os
import datetime
import torch
from torch.utils.data import DataLoader

from Yolov3 import YoloV3, Yololoss
from Yolov3 import anchors_wh, anchors_wh_mask
from preprocess import CustomDataset


BATCH_SIZE = 1
EPOCH = 1000


def main():
    # DB_path = './data/VOC2007_trainval/'
    # csv_file = '2007_train.csv'

    DB_path = './data/ex'
    csv_file = 'ex_train.csv'

    pth_dir = './models/'

    # # 학습했던 모델 불러오기
    saved_pth_dir = './models'
    pth_file = '99_2472.1938.pth'

    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda' if torch.cuda.is_available() else torch.device('cpu'))

    num_classes = 20
    lr_rate = 0.001
    pre_train = False

    dataset = CustomDataset(DB_path, csv_file, num_classes)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

    model = YoloV3(num_classes).to(device)
    loss_object = [Yololoss(num_classes, valid_anchors_wh) for valid_anchors_wh in anchors_wh_mask]
    optimizer = torch.optim.Adam(model.parameters(), lr=lr_rate)

    if pre_train:
        state = torch.load(os.path.join(saved_pth_dir, pth_file))
        model.load_state_dict(state['model_state_dict'])
        optimizer.load_state_dict(state['optimizer_state_dict'])
        init_epoch = state['epoch']
        lowest_loss = state['loss']
    else:
        init_epoch = 0
        lowest_loss = 0.1

    # print(f'init_epoch: {init_epoch}')
    # print(f'lowest_loss: {lowest_loss}')

    for epoch in range(init_epoch, EPOCH):
        model.train()
        print(f'{epoch} epoch start! : {datetime.datetime.now().strftime("%Y.%m.%d %H:%M:%S")}')

        epoch_total_loss = train_one_epoch(model, loss_object, dataloader, optimizer, use_cuda)

        if epoch_total_loss < lowest_loss:
            lowest_loss = epoch_total_loss
            state = {'epoch': epoch,
                     'model_state_dict': model.state_dict(),
                     'optimizer_state_dict': optimizer.state_dict(),
                     'loss': epoch_total_loss}
            file_path = pth_dir + f'{epoch}_{lowest_loss:.4f}.pth'
            torch.save(state, file_path)
            print(f'Save model_ [loss : {lowest_loss:.4f}, save_path : {file_path}]')

        if lowest_loss < 0.00001:
            break

    state = {'epoch': epoch,
             'model_state_dict': model.state_dict(),
             'optimizer_state_dict': optimizer.state_dict(),
             'loss': epoch_total_loss}
    file_path = pth_dir + f'{epoch}_{epoch_total_loss:.4f}.pth'
    torch.save(state, file_path)
    print(f'Save model_ [loss : {epoch_total_loss:.4f}, save_path : {file_path}]')


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

        for loss_object, y_pred, y_true in zip(loss_object, model_output, label):
            total_loss, each_loss = loss_object(y_true, y_pred)
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





