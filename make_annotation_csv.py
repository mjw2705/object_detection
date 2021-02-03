import csv
import cv2
import xml.etree.ElementTree as ET


DB_path = './data/VOC2007_trainval'
classes = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]

for set in ['train', 'val']:
    with open(f'{DB_path}/ImageSets/Main/{set}.txt', 'r') as f:
        file_id = f.read().strip().split()

    f = open(f'2007_{set}.csv', 'w', encoding='utf-8', newline='')
    writer = csv.writer(f)
    writer.writerow(['img_path', 'Bbox', 'class'])

    for fids in file_id:
        tree = ET.parse(f'{DB_path}/Annotations/{fids}.xml')
        root = tree.getroot()

        str_bboxs, str_ids = '', ''
        bboxs, ids = [], []

        for obj in root.iter('object'):
            cls = obj.find('name').text
            if cls not in classes:
                continue
            cls_id = classes.index(cls)
            bndbox = obj.find('bndbox')
            xmin = int(bndbox.find('xmin').text)
            ymin = int(bndbox.find('ymin').text)
            xmax = int(bndbox.find('xmax').text)
            ymax = int(bndbox.find('ymax').text)

            bboxs.append(f'{xmin} {ymin} {xmax} {ymax}')
            ids.append(f'{cls_id}')

        str_bboxs = ','.join(bboxs)
        str_ids = ','.join(ids)
        writer.writerow([f'JPEGImages/{fids}.jpg', str_bboxs, str_ids])

    f.close()


# # bbox 이미지에 뿌리기
# id = 'JPEGImages/000017.jpg'
# img = cv2.imread(f'{DB_path}/{id}')
#
# bnd = '185 62 279 199,90 78 403 336'
# cls = '14,12'
# bnd = list(map(str, bnd.split(',')))
# cls = list(map(int, cls.split(',')))
#
# for bbox in bnd:
#     box = list(map(int, bbox.split(' ')))
#     img = cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), (255, 0, 0), 2)
#
# cv2.imshow('t', img)
# cv2.waitKey()
# cv2.destroyAllWindows()