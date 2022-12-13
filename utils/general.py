from typing import Tuple, List
from PIL import Image, ImageDraw

import numpy as np
import torch

EPS: float = 1e-10


def draw_bboxes(img: ImageDraw, bboxes: np.ndarray) -> ImageDraw:
    """
    draw bounding boxes on the give ImageDraw
    """
    for bbox in bboxes:
        x1, y1, x2, y2 = bbox
        img.rectangle((x1, y1, x2, y2))


def get_area(bboxes: torch.Tensor) -> torch.Tensor:
    """
    calculates area of bounding boxes
    :param bboxes: a set of bounding boxes in 2d tensor (x1, y1, x2, y2)
    :return: areas in 2d tensor
    """
    width: torch.Tensor = bboxes[:, 2] - bboxes[:, 0]
    height: torch.Tensor = bboxes[:, 3] - bboxes[:, 1]

    return (width * height).unsqueeze(1)


def get_intersection(box: torch.Tensor, boxes: torch.Tensor) -> torch.Tensor:
    """
    calculates intersection of two sets of bounding boxes
    :param box: bounding box in 1d tensors (x1, y1, x2, y2)
    :param boxes: a set of bounding boxes in 2d tensors (x1, y1, x2, y2)
    :return: intersections in 1d tensor
    """
    with torch.no_grad():
        box.unsqueeze_(0)
        width: torch.Tensor = torch.min(box[:, 2], boxes[:, 2]) - torch.max(box[:, 0], boxes[:, 0])
        height: torch.Tensor = torch.min(box[:, 3], boxes[:, 3]) - torch.max(box[:, 1], boxes[:, 1])

    return torch.clamp(width * height, 0, None)


def get_ious(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """
    calculates IoUs of two sets of bounding boxes
    :param boxes1: a set of bounding boxes in 2d tensors [num bboxes (n), 4] (x1, y1, x2, y2)
    :param boxes2: a set of bounding boxes in 2d tensors [num bboxes (m), 4] (x1, y1, x2, y2)
    :return: a set of IoUs in 2d tensor (n, m)
    """
    with torch.no_grad():
        out: torch.Tensor = torch.zeros((boxes1.shape[0], boxes2.shape[0]))  # (n, m)
        box2_area: torch.Tensor = get_area(boxes2)  # (m, 1)
        for n_idx in range(boxes1.shape[0]):
            intersections: torch.Tensor = get_intersection(boxes1[n_idx], boxes2)  # (m)
            box1_area: torch.Tensor = get_area(boxes1[n_idx: n_idx + 1])  # (1, 1)
            union: torch.Tensor = ((box1_area + box2_area).squeeze() - intersections) + EPS  # (m)
            out[n_idx] = intersections / union

    return out


# def calc_grid_offset(norm_x: torch.float, norm_y: torch.float, img_w: int = 224, img_h: int = 224):
#     x: int = (norm_x * img_w).to(dtype=int)
#     y: int = (norm_y * img_h).to(dtype=int)
#
#
# a = torch.tensor([[0, 0, 1, 1]])
# b = torch.tensor([[0.5, 0.5, 1.5, 1.5]])
#
#
def xywh2xyxy(boxes: torch.Tensor) -> torch.Tensor:
    """
    turns xywh coordinates into xyxy coordinates
    :param boxes: a set of box coordinates in 2d tensor in xywh formats [x, y, w, h]
    :return: a set of box coordinates in 2d tensor in xyxy formats
    """
    x1: torch.Tensor = torch.clamp(boxes[:, 0] - boxes[:, 2] / 2, 0, 1)
    y1: torch.Tensor = torch.clamp(boxes[:, 1] - boxes[:, 3] / 2, 0, 1)
    x2: torch.Tensor = torch.clamp(boxes[:, 0] + boxes[:, 2] / 2, 0, 1)
    y2: torch.Tensor = torch.clamp(boxes[:, 1] + boxes[:, 3] / 2, 0, 1)

    return torch.stack((x1, y1, x2, y2), dim=1)


def xywhn2xyxy(boxes: np.ndarray, w: int, h: int, dw: int, dh: int) -> np.ndarray:
    """
    turns normalized xywh coordinates into xyxy coordinates with padding
    :param boxes: a set of box coordinates in 2d tensor in normalized xywh formats [x, y, w, h]
    :return: a set of box coordinates in 2d tensor in xyxy formats
    """
    y = np.zeros_like(boxes)
    y[:, 0] = (boxes[:, 0] - boxes[:, 2] / 2) * w + dw
    y[:, 1] = (boxes[:, 1] - boxes[:, 3] / 2) * h + dh
    y[:, 2] = (boxes[:, 0] + boxes[:, 2] / 2) * w + dw
    y[:, 3] = (boxes[:, 1] + boxes[:, 3] / 2) * h + dh

    return y


def xywh2xywhp(boxes: np.ndarray, h_offset: float, w_offset: float, rh: float, rw: float) -> np.ndarray:
    y = np.copy(boxes)
    y[:, 0] = (y[:, 0] + w_offset) * rw
    y[:, 1] = (y[:, 1] + h_offset) * rh
    y[:, 2] *= rw
    y[:, 3] *= rh
    return y


def scale_boxes(outs: List[List[List[float]]], shapes: Tuple[Tuple[float, float, float, float]]) -> List[List[float]]:
    scaled_boxes = []
    for out, shape in zip(outs, shapes):  # outs = [num_preds, 6] [cls_id, conf, x1, y1, x2, y2] in pixel values
        h0, w0, dh, dw = shape
        scaled_box = []
        for box in out:
            cls_id, conf, x1, y1, x2, y2 = box
            x1 = (x1 + dw) * (w0 / (w0 - dw))
            y1 = (y1 + dh) * (h0 / (h0 - dh))
            x2 = (x2 + dw) * (w0 / (w0 - dw))
            y2 = (y2 + dh) * (h0 / (h0 - dh))
            scaled_box.append([cls_id, conf, x1, y1, x2, y2])
        scaled_boxes.append(scaled_box)

    return scaled_boxes


def get_responsible_boxes(grid_offsets, pred, label) -> torch.Tensor:
    out: torch.Tensor = torch.zeros((label.shape[0], 1))

    for idx, ((j, i), target) in enumerate(zip(grid_offsets, label)):
        first_box_iou = get_ious(xywh2xyxy(pred[1:5, i, j].unsqueeze(0)), target[1:].unsqueeze(0))  # [1, 1]
        second_box_iou = get_ious(xywh2xyxy(pred[6:10, i, j].unsqueeze(0)), target[1:].unsqueeze(0))
        out[idx] = torch.argmax(torch.concat((first_box_iou, second_box_iou))).item()

    return out.to(torch.int64)


def decode(preds, conf_thres, output) -> List[torch.Tensor]:
    """
    decode yolo output
    :param preds:
    :param conf_thres:

    :return: a list of list that contains bounding boxes [class, conf, x1, y1, x2, y2]
    """
    out = []  # stores decoded information [num_detected, 6] (class, conf, x1, y1, x2, y2)
    out2 = []  # stores decoded bboxes respect to original input image resolution

    if output == 'grid':
        cell_size = 1 / 7
        for batch_i, pred in enumerate(preds):
            batch_out = []
            # original_h, original_w = shapes[batch_i]
            first_pred_mask = pred[0, :, :] > conf_thres
            first_i_idxs, first_j_idxs = torch.where(first_pred_mask == True)
            second_pred_mask = pred[5, :, :] > conf_thres
            second_i_idxs, second_j_idxs = torch.where(second_pred_mask == True)

            first_bbox_mask = first_pred_mask.repeat(5, 1, 1)
            second_bbox_mask = second_pred_mask.repeat(5, 1, 1)
            first_cls_mask = first_pred_mask.repeat(20, 1, 1)
            second_cls_mask = second_pred_mask.repeat(20, 1, 1)

            first_box_preds = pred[:5, :, :][first_bbox_mask].view(5, -1)  # [5, num_preds]
            second_box_preds = pred[5:10, :, :][second_bbox_mask].view(5, -1)  # [5, num_preds]
            first_cls_preds = pred[10:, :, :][first_cls_mask].view(20, -1)  # [20, num_preds]
            second_cls_preds = pred[10:, :, :][second_cls_mask].view(20, -1)  # [20, num_preds]

            first_cls_p, first_cls = torch.max(first_cls_preds, 0)  # [num_preds]
            second_cls_p, second_cls = torch.max(second_cls_preds, 0)

            for box_pred, cls_p, cls_idx, i_idx, j_idx in zip(
                    first_box_preds.t().tolist() + second_box_preds.t().tolist(),
                    first_cls_p.tolist() + second_cls_p.tolist(),
                    first_cls.tolist() + second_cls.tolist(),
                    first_i_idxs.tolist() + second_i_idxs.tolist(),
                    first_j_idxs.tolist() + second_j_idxs.tolist()):
                obj, x, y, w, h = box_pred
                conf = cls_p * obj
                x_c = j_idx * cell_size + x * cell_size
                y_c = i_idx * cell_size + y * cell_size
                img_w = w * 448
                img_h = h * 448
                # original_img_w = w * original_w
                # original_img_h = h * original_h

                x1 = x_c * 448 - img_w / 2
                y1 = y_c * 448 - img_h / 2
                x2 = x_c * 448 + img_w / 2
                y2 = y_c * 448 + img_h / 2

                # xx1 = x_c * original_w - original_img_w / 2
                # yy1 = y_c * original_h - original_img_h / 2
                # xx2 = x_c * original_w + original_img_w / 2
                # yy2 = y_c * original_h + original_img_h / 2

                # out.append([cls_idx, conf, x1, y1, x2, y2])
                batch_out.append([cls_idx, conf, x1, y1, x2, y2])
                # out2.append([cls_idx, conf, xx1, yy1, xx2, yy2])

            out.append(batch_out)

            # return out
    # elif output == 'anchor_box':
    #     pass
    # else:
    #     raise Exception

    return out
    # return batch_out, out2


def non_max_suppression(preds, conf_thres=0.25, iou_thres=0.45, output='grid', agnostic='True'):
    """
    :return: a set of detected objects [num bboxes, 6] (class, conf, x1, y1, x2, y2)
    """
    bboxes = decode(preds, conf_thres, output)
    return bboxes
    out = []
    out2 = []

    sort_fn = lambda box: box[1]
    sorted_bboxes = torch.tensor(sorted(bboxes, key=sort_fn, reverse=True))  # [num_bboxes, 6]
    sorted_bboxes2 = torch.tensor(sorted(bboxes, key=sort_fn, reverse=True))

    while sorted_bboxes.numel():
        bbox = sorted_bboxes[:1, :]  # [1, 6]
        if not sorted_bboxes[1:, :].numel():
            break

        sorted_bboxes = sorted_bboxes[1:, :]  # [num_left_boxes, 6]

        if agnostic:
            ious = get_ious(bbox[:, 2:], sorted_bboxes[:, 2:]).squeeze(
                0)  # (1, 4) (num_left_boxes, 4) -> (1, num_left_boxes)
            idxs = ious < iou_thres
            sorted_bboxes = sorted_bboxes[idxs, :]  # ()

            out.append(bbox.squeeze(0).tolist())

    while sorted_bboxes2.numel():
        bbox = sorted_bboxes2[:1, :]  # [1, 6]
        if not sorted_bboxes2[1:, :].numel():
            break

        sorted_bboxes2 = sorted_bboxes2[1:, :]  # [num_left_boxes, 6]

        if agnostic:
            ious = get_ious(bbox[:, 2:], sorted_bboxes2[:, 2:]).squeeze(
                0)  # (1, 4) (num_left_boxes, 4) -> (1, num_left_boxes)
            idxs = ious < iou_thres
            sorted_bboxes2 = sorted_bboxes2[idxs, :]  # ()

            out2.append(bbox.squeeze(0).tolist())

    return out, out2


