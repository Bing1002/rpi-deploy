# 
# nanodet inference in python using ncnn model 
# 


import os 
import math 

import numpy as np 
import cv2

import ncnn
import torch
import torch.nn.functional as F
from torchvision.ops.boxes import nms


_COLORS = (
    np.array(
        [
            0.000,
            0.447,
            0.741,
            0.850,
            0.325,
            0.098,
            0.929,
            0.694,
            0.125,
            0.494,
            0.184,
            0.556,
            0.466,
            0.674,
            0.188,
            0.301,
            0.745,
            0.933,
            0.635,
            0.078,
            0.184,
            0.300,
            0.300,
            0.300,
            0.600,
            0.600,
            0.600,
            1.000,
            0.000,
            0.000,
            1.000,
            0.500,
            0.000,
            0.749,
            0.749,
            0.000,
            0.000,
            1.000,
            0.000,
            0.000,
            0.000,
            1.000,
            0.667,
            0.000,
            1.000,
            0.333,
            0.333,
            0.000,
            0.333,
            0.667,
            0.000,
            0.333,
            1.000,
            0.000,
            0.667,
            0.333,
            0.000,
            0.667,
            0.667,
            0.000,
            0.667,
            1.000,
            0.000,
            1.000,
            0.333,
            0.000,
            1.000,
            0.667,
            0.000,
            1.000,
            1.000,
            0.000,
            0.000,
            0.333,
            0.500,
            0.000,
            0.667,
            0.500,
            0.000,
            1.000,
            0.500,
            0.333,
            0.000,
            0.500,
            0.333,
            0.333,
            0.500,
            0.333,
            0.667,
            0.500,
            0.333,
            1.000,
            0.500,
            0.667,
            0.000,
            0.500,
            0.667,
            0.333,
            0.500,
            0.667,
            0.667,
            0.500,
            0.667,
            1.000,
            0.500,
            1.000,
            0.000,
            0.500,
            1.000,
            0.333,
            0.500,
            1.000,
            0.667,
            0.500,
            1.000,
            1.000,
            0.500,
            0.000,
            0.333,
            1.000,
            0.000,
            0.667,
            1.000,
            0.000,
            1.000,
            1.000,
            0.333,
            0.000,
            1.000,
            0.333,
            0.333,
            1.000,
            0.333,
            0.667,
            1.000,
            0.333,
            1.000,
            1.000,
            0.667,
            0.000,
            1.000,
            0.667,
            0.333,
            1.000,
            0.667,
            0.667,
            1.000,
            0.667,
            1.000,
            1.000,
            1.000,
            0.000,
            1.000,
            1.000,
            0.333,
            1.000,
            1.000,
            0.667,
            1.000,
            0.333,
            0.000,
            0.000,
            0.500,
            0.000,
            0.000,
            0.667,
            0.000,
            0.000,
            0.833,
            0.000,
            0.000,
            1.000,
            0.000,
            0.000,
            0.000,
            0.167,
            0.000,
            0.000,
            0.333,
            0.000,
            0.000,
            0.500,
            0.000,
            0.000,
            0.667,
            0.000,
            0.000,
            0.833,
            0.000,
            0.000,
            1.000,
            0.000,
            0.000,
            0.000,
            0.167,
            0.000,
            0.000,
            0.333,
            0.000,
            0.000,
            0.500,
            0.000,
            0.000,
            0.667,
            0.000,
            0.000,
            0.833,
            0.000,
            0.000,
            1.000,
            0.000,
            0.000,
            0.000,
            0.143,
            0.143,
            0.143,
            0.286,
            0.286,
            0.286,
            0.429,
            0.429,
            0.429,
            0.571,
            0.571,
            0.571,
            0.714,
            0.714,
            0.714,
            0.857,
            0.857,
            0.857,
            0.000,
            0.447,
            0.741,
            0.314,
            0.717,
            0.741,
            0.50,
            0.5,
            0,
        ]
    )
    .astype(np.float32)
    .reshape(-1, 3)
)


class_names = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
              'train', 'truck', 'boat', 'traffic_light', 'fire_hydrant',
              'stop_sign', 'parking_meter', 'bench', 'bird', 'cat', 'dog',
              'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
              'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
              'skis', 'snowboard', 'sports_ball', 'kite', 'baseball_bat',
              'baseball_glove', 'skateboard', 'surfboard', 'tennis_racket',
              'bottle', 'wine_glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
              'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
              'hot_dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
              'potted_plant', 'bed', 'dining_table', 'toilet', 'tv', 'laptop',
              'mouse', 'remote', 'keyboard', 'cell_phone', 'microwave',
              'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock',
              'vase', 'scissors', 'teddy_bear', 'hair_drier', 'toothbrush']


def get_single_level_center_priors(batch_size, featmap_size, stride, dtype, device):
    h, w = featmap_size
    x_range = (torch.arange(w, dtype=dtype, device=device)) * stride
    y_range = (torch.arange(h, dtype=dtype, device=device)) * stride
    y, x = torch.meshgrid(y_range, x_range)
    y = y.flatten()
    x = x.flatten()
    strides = x.new_full((x.shape[0],), stride)
    proiors = torch.stack([x, y, strides, strides], dim=-1)
    return proiors.unsqueeze(0).repeat(batch_size, 1, 1)


def distance2bbox(points, distance, max_shape=None):
    """Decode distance prediction to bounding box.

    Args:
        points (Tensor): Shape (n, 2), [x, y].
        distance (Tensor): Distance from the given point to 4
            boundaries (left, top, right, bottom).
        max_shape (tuple): Shape of the image.

    Returns:
        Tensor: Decoded bboxes.
    """
    x1 = points[..., 0] - distance[..., 0]
    y1 = points[..., 1] - distance[..., 1]
    x2 = points[..., 0] + distance[..., 2]
    y2 = points[..., 1] + distance[..., 3]
    if max_shape is not None:
        x1 = x1.clamp(min=0, max=max_shape[1])
        y1 = y1.clamp(min=0, max=max_shape[0])
        x2 = x2.clamp(min=0, max=max_shape[1])
        y2 = y2.clamp(min=0, max=max_shape[0])
    return torch.stack([x1, y1, x2, y2], -1)


def batched_nms(boxes, scores, idxs, nms_cfg, class_agnostic=False):
    max_coordinate = boxes.max()
    offsets = idxs.to(boxes) * (max_coordinate + 1)
    boxes_for_nms = boxes + offsets[:, None]

    keep = nms(boxes_for_nms, scores, iou_threshold=0.6)
    boxes = boxes[keep]
    scores = scores[keep]

    return torch.cat([boxes, scores[:, None]], -1), keep


def multiclass_nms(multi_bboxes, multi_scores, score_thr, nms_cfg, max_num=-1, score_factors=None):
    num_classes = multi_scores.size(1) - 1
    # exclude background category
    if multi_bboxes.shape[1] > 4:
        bboxes = multi_bboxes.view(multi_scores.size(0), -1, 4)
    else:
        bboxes = multi_bboxes[:, None].expand(multi_scores.size(0), num_classes, 4)
    scores = multi_scores[:, :-1]

    # filter out boxes with low scores
    valid_mask = scores > score_thr

    # We use masked_select for ONNX exporting purpose,
    # which is equivalent to bboxes = bboxes[valid_mask]
    # we have to use this ugly code
    bboxes = torch.masked_select(
        bboxes, torch.stack((valid_mask, valid_mask, valid_mask, valid_mask), -1)
    ).view(-1, 4)
    if score_factors is not None:
        scores = scores * score_factors[:, None]
    scores = torch.masked_select(scores, valid_mask)
    labels = valid_mask.nonzero(as_tuple=False)[:, 1]

    if bboxes.numel() == 0:
        bboxes = multi_bboxes.new_zeros((0, 5))
        labels = multi_bboxes.new_zeros((0,), dtype=torch.long)

        if torch.onnx.is_in_onnx_export():
            raise RuntimeError(
                "[ONNX Error] Can not record NMS "
                "as it has not been executed this time"
            )
        return bboxes, labels
    
    dets, keep = batched_nms(bboxes, scores, labels, nms_cfg)

    if max_num > 0:
        dets = dets[:max_num]
        keep = keep[:max_num]

    return dets, labels[keep]


def warp_boxes(boxes, M, width, height):
    n = len(boxes)
    if n:
        # warp points
        xy = np.ones((n * 4, 3))
        xy[:, :2] = boxes[:, [0, 1, 2, 3, 0, 3, 2, 1]].reshape(
            n * 4, 2
        )  # x1y1, x2y2, x1y2, x2y1
        xy = xy @ M.T  # transform
        xy = (xy[:, :2] / xy[:, 2:3]).reshape(n, 8)  # rescale
        # create new boxes
        x = xy[:, [0, 2, 4, 6]]
        y = xy[:, [1, 3, 5, 7]]
        xy = np.concatenate((x.min(1), y.min(1), x.max(1), y.max(1))).reshape(4, n).T
        # clip boxes
        xy[:, [0, 2]] = xy[:, [0, 2]].clip(0, width)
        xy[:, [1, 3]] = xy[:, [1, 3]].clip(0, height)
        return xy.astype(np.float32)
    else:
        return boxes


def overlay_bbox_cv(img, dets, class_names, score_thresh):
    img = cv2.imread(img)

    all_box = []
    for label in dets:
        for bbox in dets[label]:
            score = bbox[-1]
            if score > score_thresh:
                x0, y0, x1, y1 = [int(i) for i in bbox[:4]]
                all_box.append([label, x0, y0, x1, y1, score])
    all_box.sort(key=lambda v: v[5])
    for box in all_box:
        label, x0, y0, x1, y1, score = box
        # color = self.cmap(i)[:3]
        color = (_COLORS[label] * 255).astype(np.uint8).tolist()
        text = "{}:{:.1f}%".format(class_names[label], score * 100)
        txt_color = (0, 0, 0) if np.mean(_COLORS[label]) > 0.5 else (255, 255, 255)
        font = cv2.FONT_HERSHEY_SIMPLEX
        txt_size = cv2.getTextSize(text, font, 0.5, 2)[0]
        cv2.rectangle(img, (x0, y0), (x1, y1), color, 2)

        cv2.rectangle(
            img,
            (x0, y0 - txt_size[1] - 1),
            (x0 + txt_size[0] + txt_size[1], y0 - 1),
            color,
            -1,
        )
        cv2.putText(img, text, (x0, y0 - 1), font, 0.5, txt_color, thickness=1)
    return img


class NanoDetInference:
    def __init__(self, param_path, model_path, img_size=(416, 416)):
        self.net = ncnn.Net()
        self.net.load_param(param_path)
        self.net.load_model(model_path)
        self.img_size = img_size  # (w, h)

    def preprocess(self, image_path):
        # read image
        # resize, pad, normalize, color convert, 
        # bhwc to bchw, to tensor, float32
        # cv2.resize or cv2.warpPerspective
        # img = cv2.resize(img, (416, 416))
        img = cv2.imread(image_path)
        h, w = img.shape[:2]
        M = np.array([[self.img_size[0] / w, 0, 0],
                      [0, self.img_size[1] / h, 0], 
                      [0, 0, 1]])
        img = cv2.warpPerspective(img, M, dsize=tuple(self.img_size))

        img = img.astype(np.float32) / 255.0
        img = (img - np.array([0.406, 0.456, 0.485])) / np.array([0.225, 0.224, 0.229])  # bgr 
        
        img = torch.from_numpy(img.transpose(2, 0, 1)).to(torch.float)
        img = torch.stack([img], dim=0).contiguous()
        return img

    def forward(self, in0):
        # tensor to output 
        out = []
        with self.net.create_extractor() as ex:
            ex.input("in0", ncnn.Mat(in0.squeeze(0).numpy()).clone())
            _, out0 = ex.extract("out0")
            out.append(torch.from_numpy(np.array(out0)).unsqueeze(0))

        if len(out) == 1:
            return out[0].transpose(1, 2).contiguous()
        else:
            return tuple(out)


    def postprocess(self, preds):
        # genearte grid cells
        # sort, softmax, nms 
        # convert to xywh, confidence, class
        num_classes = 80
        reg_max = 7

        cls_scores, bbox_preds = preds.split(
            [num_classes, 4 * (reg_max + 1)], dim=-1
        )

        b = cls_scores.shape[0]
        input_height, input_width = 416, 416
        input_shape = (input_height, input_width)

        strides = [8, 16, 32, 64]

        featmap_sizes = [
            (math.ceil(input_height / stride), math.ceil(input_width) / stride)
            for stride in strides
        ]


        # get grid cells of one image
        mlvl_center_priors = [
            get_single_level_center_priors(
                b,
                featmap_sizes[i],
                stride,
                dtype=torch.float32, 
                device=cls_scores.device
            )
            for i, stride in enumerate(strides)
        ]

        #  
        center_priors = torch.cat(mlvl_center_priors, dim=1)

        # 
        # This approach, known as "Distribution Focal Loss" or "DFL", represents bounding 
        # box coordinates as discrete probability distributions. The softmax operation 
        # creates these distributions, and the linear projection converts them back to 
        # continuous coordinate values. This method can potentially improve the accuracy 
        # and stability of bounding box predictions in object detection tasks
        #
        shape = bbox_preds.size()
        project = torch.linspace(0, reg_max, reg_max + 1)
        bbox_preds = F.softmax(bbox_preds.reshape(*shape[:-1], 4, reg_max + 1), dim=-1)
        bbox_preds = F.linear(bbox_preds, project.type_as(bbox_preds)).reshape(*shape[:-1], 4)

        dis_preds = bbox_preds * center_priors[..., 2, None]
        bboxes = distance2bbox(center_priors[..., :2], dis_preds, max_shape=input_shape)
        scores = cls_scores.sigmoid()

        # batch nms 
        result_list = []
        for i in range(b):
            # add a dummy background class at the end of all labels
            # same with mmdetection2.0
            score, bbox = scores[i], bboxes[i]
            padding = score.new_zeros(score.shape[0], 1)
            score = torch.cat([score, padding], dim=1)
            results = multiclass_nms(
                bbox,
                score,
                score_thr=0.05,
                nms_cfg=dict(type="nms", iou_threshold=0.6),
                max_num=100,
            )
            result_list.append(results)

        # rescale to original image size
        result = result_list[0]
        warp_matrix = np.array([[416/810, 0, 0], 
                                [0, 416/1080, 0], 
                                [0, 0, 1]])
        img_width, img_height = 810, 1080


        det_result = {}
        det_bboxes, det_labels = result
        det_bboxes = det_bboxes.detach().cpu().numpy()
        det_bboxes[:, :4] = warp_boxes(
            det_bboxes[:, :4], np.linalg.inv(warp_matrix), img_width, img_height
        )

        # output format 
        classes = det_labels.detach().cpu().numpy()
        for i in range(num_classes):
            inds = classes == i
            det_result[i] = np.concatenate(
                [
                    det_bboxes[inds, :4].astype(np.float32),
                    det_bboxes[inds, 4:5].astype(np.float32),
                ],
                axis=1,
            ).tolist()

        return det_result  


    def inference(self, image_path):
        image = self.preprocess(image_path)
        out = self.forward(image)
        result = self.postprocess(out)

        return result




if __name__ == "__main__":
    # fake input 
    torch.manual_seed(0)
    in0 = torch.rand(1, 3, 640, 640, dtype=torch.float)
    image_path = "/home/bing/code/open-source/rpi-deploy/data/bus.jpg"

    param_path = "/home/bing/code/checkpoints/nanodet/nanodet_plus_m_1.5x_416_torchscript.ncnn.param"
    model_path = "/home/bing/code/checkpoints/nanodet/nanodet_plus_m_1.5x_416_torchscript.ncnn.bin"

    model = NanoDetInference(param_path, model_path)
    result = model.inference(image_path)

    # visualize result
    result = overlay_bbox_cv(image_path, result, class_names, 
                             score_thresh=0.5)
    cv2.imwrite("result.jpg", result)
    
    print("Done")
