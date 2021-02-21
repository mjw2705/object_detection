import torch
import torch.nn as nn
from utils import broadcast_iou, xywh_to_x1x2y1y2


class Postprocess(nn.Module):
    def __init__(self, iou_threshold=0.5, score_threshold=0.5, max_detection=100):
        super(Postprocess, self).__init__()
        self.iou_threshold = iou_threshold
        self.score_threshold = score_threshold
        self.max_detection = max_detection

    def forward(self, raw_yolo_output):
        boxes, objectness, class_prob = [], [], []

        # raw_yolo_out (bbox_abs, objectness, class_probs, bbox_rel)
        for raw_yolo_out in raw_yolo_output:
            batch_size = raw_yolo_out[0].size(0)
            num_classes = raw_yolo_out[2].size(-1)

            boxes.append(raw_yolo_out[0].view(batch_size, -1, 4))
            objectness.append(raw_yolo_out[1].contiguous().view(batch_size, -1, 1))
            class_prob.append(raw_yolo_out[2].contiguous().view(batch_size, -1, num_classes))

        boxes = xywh_to_x1x2y1y2(torch.cat(boxes, dim=1))
        objectness = torch.cat(objectness, dim=1)
        class_prob = torch.cat(class_prob, dim=1)

        scores = objectness
        scores_shape = scores.shape
        scores = torch.reshape(scores, [scores_shape[0], -1, scores_shape[-1]])

        return self.batch_non_maximum_suppression(boxes, scores, class_prob)

    def batch_non_maximum_suppression(self, boxes, scores, classes):
        def single_batch_nms(candidate_boxes):
            y_mask = candidate_boxes[..., 4] >= self.score_threshold  # true or false
            candidate_boxes = candidate_boxes[y_mask]
            outputs = torch.zeros((self.max_detection + 1, candidate_boxes.size(-1)))

            indices = []
            updates = []

            count = 0
            # candidate_boxes가 없거나 max_detection을 다 채울때 까지 반복
            while candidate_boxes.size(0) > 0 and count < self.max_detection:
                # candidate_boxes 중에서 점수가 가장 높은 박스 pick
                best_idx = torch.argmax(candidate_boxes[..., 4], dim=0)
                best_box = candidate_boxes[best_idx]

                indices.append([count] * candidate_boxes.size(-1))
                updates.append(best_box)
                count += 1
                # best_box는 candidate_boxes에서 제거
                candidate_boxes = torch.cat((candidate_boxes[0:best_idx],
                                             candidate_boxes[best_idx + 1:candidate_boxes.size(0)]), dim=0)
                # best_box와 모든 candidate_boxes 비교
                iou = broadcast_iou(best_box[0:4], candidate_boxes[..., 0:4])
                # iou가 iou_threshold보다 큰 후보 상자 제거
                candidate_boxes = candidate_boxes[iou <= self.iou_threshold]

            # 한번이라도 count가 됬을 때
            if count > 0:
                count_idx = [[self.max_detection] * candidate_boxes.size(-1)]
                count_update = [torch.zeros(candidate_boxes.size(-1)).fill_(count)]
                indices = torch.cat((torch.tensor(indices), torch.tensor(count_idx)), dim=0)
                updates = torch.cat((torch.stack(updates), torch.stack(count_update).cuda()), dim=0)
                # dim=0으로 outputs의 indices위치에 updates값 넣기
                outputs = outputs.cuda().scatter_(0, indices.cuda(), updates)

            return outputs

        valid_count = []
        final_result = []
        combined_boxes = torch.cat((boxes, scores, classes), dim=2)

        for combined_box in combined_boxes:
            result = single_batch_nms(combined_box)
            valid_count.append(result[self.max_detection][0].unsqueeze(0).unsqueeze(0))
            final_result.append(result[0:self.max_detection].unsqueeze(0))

        valid_count = torch.cat(valid_count, dim=0)
        final_result = torch.cat(final_result, dim=0)

        nms_boxes, nms_scores, nms_classes = torch.split(final_result, [4, 1, final_result.size(-1) - 5], dim=-1)

        return nms_boxes, nms_scores, nms_classes, valid_count.int()
