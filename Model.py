import os
import torch
import numpy as np
import random
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
import torchvision.models.detection
from torchvision.ops import nms, RoIPool
from torch.optim.lr_scheduler import StepLR
import skimage.io as io
from typing import Union, Tuple
import pandas as pd
import cv2 as cv
import matplotlib.pyplot as plt
from matplotlib import patches
from Config import *


def nms_fun(boxes: Tensor, scores: Tensor, iou_threshold: float) -> Tensor:
    """
    非极大值抑制
    :param boxes: [n, 4]维边界框, 列坐标分别对应(x1, y1, x2, y2)
    :param scores: [n]维边界框的得分
    :param iou_threshold: 非极大值抑制过程中IoU阈值
    :return: [m]维, 保留下来的边界框序号
    """
    # 计算面积
    # # 注意, 如果要与torchvision.opt.nms结果对齐, 此处利用长宽计算面积时不加1
    # areas = (boxes[:, 3] - boxes[:, 1] + 1) * (boxes[:, 2] - boxes[:, 0] + 1)
    areas = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

    # 将得分降序排列, 获取降序索引, torch.sort会返回元组(排序后的张量, 排序索引)
    _, order = torch.sort(input=scores, dim=0, descending=True)
    keep = []

    while order.size(0) > 0:
        idx = order[0]
        keep.append(idx)

        if order.size == 1:
            break
        # 计算相交区域
        xx1 = torch.maximum(boxes[idx, 0], boxes[order[1:], 0])
        xx2 = torch.minimum(boxes[idx, 2], boxes[order[1:], 2])
        yy1 = torch.maximum(boxes[idx, 1], boxes[order[1:], 1])
        yy2 = torch.minimum(boxes[idx, 3], boxes[order[1:], 3])
        # 计算交集面积
        # # 同理, 如果要与torchvision.opt.nms结果对齐, 此处利用长宽计算面积时不加1
        # inter = torch.clip(xx2 - xx1 + 1, min=0.0) * torch.clip(yy2 - yy1 + 1, min=0.0)
        inter = torch.clip(xx2 - xx1, min=0.0) * torch.clip(yy2 - yy1, min=0.0)
        iou = inter / (areas[idx] + areas[order[1:]] - inter)
        # 保留IoU小于阈值的bbox
        idx = torch.where(iou <= iou_threshold)[0]
        order = order[idx + 1]

    keep = torch.tensor(keep)
    return keep


def backbone(pretrained=False):
    """
    定义主干特征提取网络和最终推荐区域特征处理线性网络
    :param pretrained: 是否加载预训练参数
    :return: 特征提取器和后续线性全连接层
    """
    net = torchvision.models.vgg16(pretrained=pretrained)
    extractor = net.features[:-1]
    linear = nn.Sequential(
        nn.Linear(in_features=512 * 7 * 7, out_features=4096, bias=True),
        nn.ReLU()
    )
    if not pretrained:
        extractor.apply(lambda x: nn.init.kaiming_normal_(x.weight) if isinstance(x, nn.Conv2d) else None)
        linear.apply(lambda x: nn.init.kaiming_normal_(x.weight) if isinstance(x, nn.Linear) else None)
    return extractor, linear


class AnchorCreator:
    # 生成先验框对应的标签及与真值框间的真实偏移值
    def __init__(self, num_samples=256, positive_iou_thresh=0.7, negative_iou_thresh=0.3, positive_rate=0.5):
        """
        初始化anchor生成器
        :param num_samples: 每帧图片上用于后续分类和回归任务的有效推荐区域总数
        :param positive_iou_thresh: 正样本的IoU判定阈值
        :param negative_iou_thresh: 负样本的判定阈值
        :param positive_rate: 正样本所占样本总数的比例
        """
        self.num_samples = num_samples
        self.positive_iou_thresh = positive_iou_thresh
        self.negative_iou_thresh = negative_iou_thresh
        self.positive_rate = positive_rate

    @staticmethod
    def is_inside_anchors(anchors: Union[np.ndarray, Tensor], width: int, height: int) -> Union[np.ndarray, Tensor]:
        """
        获取图像内部的推荐框
        :param anchors: 生成的所有推荐框->[x1, y1, x2, y2]
        :param width: 输入图像宽度
        :param height: 输入图像高度
        :return: 未超出图像边界的推荐框
        """
        is_inside = (anchors[:, 0] >= 0) & (anchors[:, 1] >= 0) & (anchors[:, 2] <= width - 1) & (anchors[:, 3] <= height - 1)
        return is_inside

    @staticmethod
    def calc_IoU(anchors: np.ndarray, gt_boxes: np.ndarray, method=1) -> np.ndarray:
        """
        计算推荐区域与真值的IoU
        :param anchors: 推荐区域边界框, [m, 4]维数组, 四列分别对应左上和右下两个点坐标[x1, y1, x2, y2]
        :param gt_boxes: 当前图像中所有真值边界框, [n, 4]维数组, 四列分别对应左上和右下两点坐标[x1, y1, x2, y2]
        :param method: iou计算方法
        :return: iou, [m, n]维数组, 记录每个推荐区域与每个真值框的IoU结果
        """
        # 先判断维度是否符合要求
        assert anchors.ndim == gt_boxes.ndim == 2, "anchors and ground truth bbox must be 2D array."
        assert anchors.shape[1] == gt_boxes.shape[1] == 4, "anchors and ground truth bbox must contain 4 values for 2 points."
        num_anchors, num_gts = anchors.shape[0], gt_boxes.shape[0]

        # 方法1: 利用for循环遍历求解交并比
        if method == 0:
            iou = np.zeros((num_anchors, num_gts))
            # anchor有m个, gt_box有n个, 遍历求出每个gt_box对应的iou结果即可
            for idx in range(num_gts):
                gt_box = gt_boxes[idx]
                box_area = (anchors[:, 2] - anchors[:, 0]) * (anchors[:, 3] - anchors[:, 1])
                gt_area = (gt_box[2] - gt_box[0]) * (gt_box[3] - gt_box[1])

                inter_w = np.minimum(anchors[:, 2], gt_box[2]) - np.maximum(anchors[:, 0], gt_box[0])
                inter_h = np.minimum(anchors[:, 3], gt_box[3]) - np.maximum(anchors[:, 1], gt_box[1])

                inter = np.maximum(inter_w, 0) * np.maximum(inter_h, 0)
                union = box_area + gt_area - inter
                iou[:, idx] = inter / union

        # 方法2: 利用repeat对矩阵进行升维, 从而利用对应位置计算交并比
        elif method == 1:
            # anchors维度为[m, 4], gt_boxes维度为[n, 4], 对二者通过repeat的方式都升维到[m, n, 4]
            anchors = np.repeat(anchors[:, np.newaxis, :], num_gts, axis=1)
            gt_boxes = np.repeat(gt_boxes[np.newaxis, :, :], num_anchors, axis=0)

            # 利用对应位置求解框面积
            anchors_area = (anchors[:, :, 2] - anchors[:, :, 0]) * (anchors[:, :, 3] - anchors[:, :, 1])
            gt_boxes_area = (gt_boxes[:, :, 2] - gt_boxes[:, :, 0]) * (gt_boxes[:, :, 3] - gt_boxes[:, :, 1])

            # 求交集区域的宽和高
            inter_w = np.minimum(anchors[:, :, 2], gt_boxes[:, :, 2]) - np.maximum(anchors[:, :, 0], gt_boxes[:, :, 0])
            inter_h = np.minimum(anchors[:, :, 3], gt_boxes[:, :, 3]) - np.maximum(anchors[:, :, 1], gt_boxes[:, :, 1])

            # 求交并比
            inter = np.maximum(inter_w, 0) * np.maximum(inter_h, 0)
            union = anchors_area + gt_boxes_area - inter
            iou = inter / union

        # 方法3: 利用np函数的广播机制求结果而避免使用循环
        else:
            # 计算anchors和gt_boxes左上角点的最大值, 包括两x1坐标最大值和y1坐标最大值
            # 注意anchors[:, None, :2]会增加一个新维度, 维度为[m, 1, 2], gt_boxes[:, :2]维度为[n, 2], maximum计算最大值时会将二者都扩展到[m, n, 2]
            max_left_top = np.maximum(anchors[:, None, :2], gt_boxes[:, :2])
            # 计算anchors和gt_boxes右下角点的最小值, 包括两x2坐标最大值和y2坐标最大值, 同上也用到了广播机制
            min_right_bottom = np.minimum(anchors[:, None, 2:], gt_boxes[:, 2:])

            # 求交集面积和并集面积
            # min_right_bottom - max_left_top维度为[m, n, 2], 后两列代表交集区域的宽和高
            # 用product进行两列元素乘积求交集面积, 用(max_left_top < min_right_bottom).all(axis=2)判断宽和高是否大于0, 结果维度为[m, n]
            inter = np.product(min_right_bottom - max_left_top, axis=2) * (max_left_top < min_right_bottom).all(axis=2)
            # 用product进行两列元素乘积求每个anchor的面积, 结果维度维[m]
            anchors_area = np.product(anchors[:, 2:] - anchors[:, :2], axis=1)
            # 用product进行两列元素乘积求每个gt_box的面积, 结果维度维[n]
            gt_boxes_area = np.product(gt_boxes[:, 2:] - gt_boxes[:, :2], axis=1)
            # anchors_area[:, None]维度维[m, 1], gt_boxes_area维度维[n], 二者先广播到[m, n]维度, 再和同纬度inter做减法计算, 结果维度维[m, n]
            union = anchors_area[:, None] + gt_boxes_area - inter
            iou = inter / union

        return iou

    @staticmethod
    def calc_max_iou_info(iou: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        利用iou结果计算出最大iou及其对应位置
        :param iou: [m, n]维矩阵, 其中m为anchors数量, n为gt_boxes数量
        :return: 每一列最大iou出现的行编号, 每一行最大iou出现的列编号, 每一行的最大iou结果
        """
        # 按列求每一列的iou最大值出现的行数, 即记录与每个gt_box的iou最大的anchor的行编号, 维度和gt_box个数相同, 为n(每个gt_box对应一个anchor与之iou最大)
        max_iou_idx_anchor = np.argmax(iou, axis=0)
        # 按行求每一行的iou最大值出现的列数, 即记录与每个anchor的iou最大的gt_box的列编号, 维度和anchor个数相同, 为m(每个anchor对应一个gt_box与之iou最大)
        max_iou_idx_gt = np.argmax(iou, axis=1)
        # 求每个anchor与所有gt_box的最大iou值
        max_iou_values_anchor = np.max(iou, axis=1)
        return max_iou_idx_anchor, max_iou_idx_gt, max_iou_values_anchor

    def create_anchor_labels(self, anchors: np.ndarray, gt_boxes: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        计算IoU结果并根据结果为每个推荐区域生成标签
        :param anchors: 生成的有效推荐区域, 列坐标对应[x1, y1, x2, y2]
        :param gt_boxes: 真值框, 列坐标对应[x1, y1, x2, y2]
        :return: 每个推荐区域的最大iou对应的真值框编号, 推荐区域对应的标签
        """
        # 计算iou结果
        iou = self.calc_IoU(anchors=anchors, gt_boxes=gt_boxes)
        # 计算行/列方向最大iou对应位置和值
        max_iou_idx_anchor, max_iou_idx_gt, max_iou_values_anchor = self.calc_max_iou_info(iou=iou)

        # 先将所有label置为-1, -1表示不进行处理, 既不是正样本也不是负样本, 再根据iou判定正样本为1, 背景为0
        labels = -1 * np.ones(anchors.shape[0], dtype="int")
        # max_iou_values_anchor为每一行最大的iou结果, 其值低于负样本阈值, 表明该行对应的anchor与所有gt_boxes的iou结果均小于阈值, 设置为负样本
        labels[max_iou_values_anchor < self.negative_iou_thresh] = 0
        # max_iou_idx_anchor为每一列iou最大值出现的行编号, 表明对应行anchor与某个gt_box的iou最大, iou最大肯定是设置为正样本
        labels[max_iou_idx_anchor] = 1
        # max_iou_values_anchor为每一行最大的iou结果, 其值大于正样本阈值, 表明该行对应的anchor与至少一个gt_box的iou结果大于阈值, 设置为正样本
        labels[max_iou_values_anchor >= self.positive_iou_thresh] = 1

        # 对正负样本数量进行限制
        # 计算目标正样本数量
        num_positive = int(self.num_samples * self.positive_rate)
        # 记录正样本行编号
        idx_positive = np.where(labels == 1)[0]
        if len(idx_positive) > num_positive:
            size_to_rest = len(idx_positive) - num_positive
            # 从正样本编号中随机选取一定数量将标签置为-1
            idx_to_reset = np.random.choice(a=idx_positive, size=size_to_rest, replace=False)
            labels[idx_to_reset] = -1

        # 计算现有负样本数量
        num_negative = self.num_samples - np.sum(labels == 1)
        # 记录负样本行编号
        idx_negative = np.where(labels == 0)[0]
        if len(idx_negative) > num_negative:
            size_to_rest = len(idx_negative) - num_negative
            # 从负样本编号中随机选取一定数量将标签置为-1
            idx_to_reset = np.random.choice(a=idx_negative, size=size_to_rest, replace=False)
            labels[idx_to_reset] = -1

        return max_iou_idx_gt, labels

    @staticmethod
    def calc_offsets_from_bboxes(anchors: np.ndarray, target_boxes: np.ndarray, eps: float = 1e-5) -> np.ndarray:
        """
        计算推荐区域与真值间的位置偏移
        :param anchors: 候选边界框, 列坐标对应[x1, y1, x2, y2]
        :param target_boxes: 真值, 列坐标对应[x1, y1, x2, y2]
        :param eps: 极小值, 防止除以0或者负数
        :return: 边界框偏移值->[dx, dy, dw, dh]
        """
        offsets = np.zeros_like(anchors, dtype="float32")
        # 计算anchor中心点坐标及长宽
        anchors_h = anchors[:, 3] - anchors[:, 1] + 1
        anchors_w = anchors[:, 2] - anchors[:, 0] + 1
        anchors_cy = 0.5 * (anchors[:, 3] + anchors[:, 1])
        anchors_cx = 0.5 * (anchors[:, 2] + anchors[:, 0])

        # 计算目标真值框中心点坐标及长宽
        targets_h = target_boxes[:, 3] - target_boxes[:, 1] + 1
        targets_w = target_boxes[:, 2] - target_boxes[:, 0] + 1
        targets_cy = 0.5 * (target_boxes[:, 3] + target_boxes[:, 1])
        targets_cx = 0.5 * (target_boxes[:, 2] + target_boxes[:, 0])

        # 限制anchor长宽防止小于0
        anchors_w = np.maximum(anchors_w, eps)
        anchors_h = np.maximum(anchors_h, eps)
        # 计算偏移值
        offsets[:, 0] = (targets_cx - anchors_cx) / anchors_w
        offsets[:, 1] = (targets_cy - anchors_cy) / anchors_h
        offsets[:, 2] = np.log(targets_w / anchors_w)
        offsets[:, 3] = np.log(targets_h / anchors_h)

        return offsets

    def __call__(self, im_width: int, im_height: int, anchors: np.ndarray, gt_boxes: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        利用真值框和先验框的iou结果为每个先验框打标签, 同时计算先验框和真值框对应的偏移值
        :param im_width: 输入图像宽度
        :param im_height: 输入图像高度
        :param anchors: 全图先验框, 列坐标对应[x1, y1, x2, y2]
        :param gt_boxes: 真值框, 列坐标对应[x1, y1, x2, y2]
        :return: 先验框对应标签和应该产生的偏移值
        """
        num_anchors = len(anchors)
        # 获取有效的推荐区域, 其维度为[m], m <= num_anchors
        is_inside = self.is_inside_anchors(anchors=anchors, width=im_width, height=im_height)
        inside_anchors = anchors[is_inside]

        # 在有效先验框基础上, 获取每个先验框的最大iou对应的真值框编号和区域标签
        max_iou_idx_gt, inside_labels = self.create_anchor_labels(anchors=inside_anchors, gt_boxes=gt_boxes)

        # 每个anchor都存在n个真值框, 选择最大iou对应的那个真值框作为每个anchor的目标框计算位置偏移
        # gt_boxes维度为[n, 4], max_iou_idx_gt维度为[m], 从真值中挑选m次即得到与每个anchor的iou最大的真值框, 即所需目标框, 维度为[m, 4]
        target_boxes = gt_boxes[max_iou_idx_gt]
        inside_offsets = self.calc_offsets_from_bboxes(anchors=inside_anchors, target_boxes=target_boxes)

        # 上面的偏移值和labels都是在inside_anchors中求得, 现在将结果映射回全图
        # 将所有标签先置为-1, 再将内部先验框标签映射回全图
        labels = -1 * np.ones(num_anchors)
        labels[is_inside] = inside_labels
        # 将所有偏移值先置为0, 再将内部先验框偏移值映射回全图
        offsets = np.zeros_like(anchors)
        offsets[is_inside] = inside_offsets

        return labels, offsets


class ProposalCreator:
    # 对每一幅图像, 利用偏移值对所有先验框进行位置矫正得到目标建议框, 再通过尺寸限制/得分限制/nms方法对目标建议框进行过滤, 获得推荐区域, 即每幅图像的roi区域
    def __init__(self, nms_thresh=0.7, num_samples_train=(12000, 2000), num_samples_test=(6000, 300), min_size=16, train_flag=False):
        """
        初始化推荐区域生成器, 为每幅图像生成满足尺寸要求、得分要求、nms要求的规定数量推荐框
        :param nms_thresh: 非极大值抑制阈值
        :param num_samples_train: 训练过程非极大值抑制前后待保留的样本数
        :param num_samples_test: 测试过程非极大值抑制步骤前后待保留的样本数
        :param min_size: 边界框最小宽高限制
        :param train_flag: 模型训练还是测试
        """
        self.train_flag = train_flag
        self.nms_thresh = nms_thresh
        self.num_samples_train = num_samples_train
        self.num_samples_test = num_samples_test
        self.min_size = min_size
    
    @staticmethod
    def calc_bboxes_from_offsets(offsets: Tensor, anchors: Tensor, eps=1e-5) -> Tensor:
        """
        由图像特征计算的偏移值offsets对rpn产生的先验框位置进行修正
        :param offsets: 偏移值矩阵->[n, 4], 列对应[x1, y1, x2, y2]的偏移值
        :param anchors: 先验框矩阵->[n, 4], 列坐标对应[x1, y1, x2, y2]
        :param eps: 极小值, 防止乘以0或者负数
        :return: 目标坐标矩阵->[n, 4], 对应[x1, y1, x2, y2]
        """
        eps = torch.tensor(eps).type_as(offsets)
        targets = torch.zeros_like(offsets, dtype=torch.float32)
        # 计算目标真值框中心点坐标及长宽
        anchors_h = anchors[:, 3] - anchors[:, 1] + 1
        anchors_w = anchors[:, 2] - anchors[:, 0] + 1
        anchors_cx = 0.5 * (anchors[:, 2] + anchors[:, 0])
        anchors_cy = 0.5 * (anchors[:, 1] + anchors[:, 3])

        anchors_w = torch.maximum(anchors_w, eps)
        anchors_h = torch.maximum(anchors_h, eps)

        # 将偏移值叠加到真值上计算anchors的中心点和宽高
        targets_w = anchors_w * torch.exp(offsets[:, 2])
        targets_h = anchors_h * torch.exp(offsets[:, 3])
        targets_cx = anchors_cx + offsets[:, 0] * anchors_w
        targets_cy = anchors_cy + offsets[:, 1] * anchors_h

        targets[:, 0] = targets_cx - 0.5 * (targets_w - 1)
        targets[:, 1] = targets_cy - 0.5 * (targets_h - 1)
        targets[:, 2] = targets_cx + 0.5 * (targets_w - 1)
        targets[:, 3] = targets_cy + 0.5 * (targets_h - 1)
        return targets

    def __call__(self, offsets: Tensor, anchors: Tensor, scores: Tensor, im_size: tuple, scale: float = 1.0) -> Tensor:
        """
        利用回归器偏移值/全图先验框/分类器得分生成满足条件的推荐区域
        :param offsets: 偏移值->[fw * fh * num_anchor_base, 4], 列坐标对应[x1, y1, x2, y2]
        :param anchors: 全图先验框->[fw * fh * num_anchor_base, 4], 列坐标对应[x1, y1, x2, y2]
        :param scores: 目标得分->[fw * fh * num_anchor_base]
        :param im_size: 原始输入图像大小 (im_height, im_width)
        :param scale: scale和min_size一起控制先验框最小尺寸
        :return: 经过偏移值矫正及过滤后保留的目标建议框, 维度[num_samples_after_nms, 4]列坐标对应[x1, y1, x2, y2]
        """
        # 设置nms过程前后需保留的样本数量, 注意训练和测试过程保留的样本数量不一致
        if self.train_flag:
            num_samples_before_nms, num_samples_after_nms = self.num_samples_train
        else:
            num_samples_before_nms, num_samples_after_nms = self.num_samples_test

        im_height, im_width = im_size
        # 利用回归器计算的偏移值对全图先验框位置进行矫正, 获取矫正后的目标先验框坐标
        targets = self.calc_bboxes_from_offsets(offsets=offsets, anchors=anchors)

        # 对目标目标先验框坐标进行限制, 防止坐标落在图像外
        # 保证0 <= [x1, x2] <= cols - 1
        targets[:, [0, 2]] = torch.clip(targets[:, [0, 2]], min=0, max=im_width - 1)
        # 0 <= [y1, y2] <= rows - 1
        targets[:, [1, 3]] = torch.clip(targets[:, [1, 3]], min=0, max=im_height - 1)

        # 利用min_size和scale控制先验框尺寸下限, 移除尺寸太小的目标先验框
        min_size = self.min_size * scale
        # 计算目标先验框宽高
        targets_w = targets[:, 2] - targets[:, 0] + 1
        targets_h = targets[:, 3] - targets[:, 1] + 1
        # 根据宽高判断框是否有效, 挑选出有效边界框和得分
        is_valid = (targets_w >= min_size) & (targets_h >= min_size)
        targets = targets[is_valid]
        scores = scores[is_valid]

        # 利用区域目标得分对目标先验框数量进行限制
        # 对目标得分进行降序排列, 获取降序索引
        descend_order = torch.argsort(input=scores, descending=True)
        # 在nms之前, 选取固定数量得分稍高的目标先验框
        if num_samples_before_nms > 0:
            descend_order = descend_order[:num_samples_before_nms]
        targets = targets[descend_order]
        scores = scores[descend_order]

        # 利用非极大值抑制限制边界框数量
        keep = nms(boxes=targets, scores=scores, iou_threshold=self.nms_thresh)

        # 如果数量不足则随机抽取, 用于填补不足
        if len(keep) < num_samples_after_nms:
            random_indexes = np.random.choice(a=range(len(keep)), size=(num_samples_after_nms - len(keep)), replace=True)
            keep = torch.concat([keep, keep[random_indexes]])

        # 在nms后, 截取固定数量边界框, 即最终生成的roi区域
        keep = keep[:num_samples_after_nms]
        targets = targets[keep]

        return targets


class RegionProposalNet(nn.Module):
    # rpn推荐区域生成网络, 获取由图像特征前向计算得到的回归器偏移值/分类器结果, 以及经过偏移值矫正后的roi区域/roi对应数据索引/全图先验框
    def __init__(self, in_channels=512, mid_channels=512, feature_stride=16, wh_ratios=(0.5, 1.0, 2.0), anchor_spatial_scales=(8, 16, 32), train_flag=False):
        super(RegionProposalNet, self).__init__()
        # 特征点步长, 即特征提取网络的空间尺度缩放比例, 工程中vgg16网络使用了4层2*2的MaxPool, 故取值为16
        self.feature_stride = feature_stride

        # 推荐框生成器: 对全图先验框进行位置矫正和过滤, 提取位置矫正后的固定数量的目标推荐框
        self.create_proposals = ProposalCreator(train_flag=train_flag)

        # 根据宽高比和空间尺度比例, 生成固定数量的基础先验框
        self.anchor_base = self.generate_anchor_base(wh_ratios=wh_ratios, anchor_spatial_scales=anchor_spatial_scales)
        num_anchor_base = len(wh_ratios) * len(anchor_spatial_scales)

        # rpn网络的3*3卷积, 目的是对输入feature map进行卷积, 进一步集中特征信息
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=mid_channels, kernel_size=(3, 3), stride=(1, 1), padding=1)
        # rpn网络分类分支, 逐像素对特征图上anchor进行分类(每个anchor对应前景/背景两类, 每个类别为一个通道, 总通道数为num_anchor_base * 2)
        self.classifier = nn.Conv2d(in_channels=mid_channels, out_channels=num_anchor_base * 2, kernel_size=(1, 1), stride=(1, 1), padding=0)
        # rpn网络回归分支, 逐像素对特征图上anchor进行坐标回归(每个框对应四个坐标值, 每个坐标为一个通道, 总通道数为num_anchor_base * 4)
        self.regressor = nn.Conv2d(in_channels=mid_channels, out_channels=num_anchor_base * 4, kernel_size=(1, 1), stride=(1, 1), padding=0)

        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)

        # 权重初始化
        nn.init.kaiming_normal_(self.conv.weight)
        nn.init.constant_(self.conv.bias, 0.0)
        nn.init.kaiming_normal_(self.classifier.weight)
        nn.init.constant_(self.classifier.bias, 0.0)
        nn.init.kaiming_normal_(self.regressor.weight)
        nn.init.constant_(self.regressor.bias, 0.0)

    @staticmethod
    def generate_anchor_base(base_size=16, wh_ratios=(0.5, 1.0, 2.0), anchor_spatial_scales=(8, 16, 32)) -> np.ndarray:
        """
        生成基础先验框->(x1, y1, x2, y2)
        :param base_size: 预设的最基本的正方形锚框边长
        :param wh_ratios: 锚框宽高比w/h取值
        :param anchor_spatial_scales: 待生成正方形框锚与预设的最基本锚框空间尺度上的缩放比例
        :return: 生成的锚框
        """
        # 默认锚框左上角为(0, 0)时计算预设锚框的中心点坐标
        cx = cy = (base_size - 1) / 2.0

        # 根据锚框宽高比取值个数M1和锚框空间尺度缩放取值数量M2, 生成M2组面积基本相同但宽高比例不同的基础锚框, 共N个（N=M1*M2）
        # 假设wh_ratios=(0.5, 1.0, 2.0)三种取值, anchor_scales=(8, 16, 32)三种取值, 那么生成的基础锚框有9种可能取值
        num_anchor_base = len(wh_ratios) * len(anchor_spatial_scales)
        # 生成[N, 4]维的基础锚框
        anchor_base = np.zeros((num_anchor_base, 4), dtype=np.float32)

        # 根据锚框面积计算锚框宽和高
        # 锚框面积s=w*h, 而wh_ration=w/h, 则s=h*h*wh_ratio, 在已知面积和宽高比时: h=sqrt(s/wh_ratio)
        # 同样可得s=w*w/wh_ratio, 在已知面积和宽高比时: w=sqrt(s*wh_ratio)

        # 计算不同宽高比、不同面积大小的锚框
        for i in range(len(wh_ratios)):
            # 遍历空间尺度缩放比例
            for j in range(len(anchor_spatial_scales)):
                # 预设框面积为s1=base_size^2, 经空间尺度缩放后为s2=(base_size*anchor_spatial_scale)^2
                # 将s2带入上述锚框宽和高计算过程可求w和h值
                h = base_size * anchor_spatial_scales[j] / np.sqrt(wh_ratios[i])
                w = base_size * anchor_spatial_scales[j] * np.sqrt(wh_ratios[i])

                idx = i * len(anchor_spatial_scales) + j

                anchor_base[idx, 0] = cx - (w - 1) / 2.0
                anchor_base[idx, 1] = cy - (h - 1) / 2.0
                anchor_base[idx, 2] = cx + (w - 1) / 2.0
                anchor_base[idx, 3] = cy + (h - 1) / 2.0
        return anchor_base

    @staticmethod
    def generate_shifted_anchors(anchor_base: np.ndarray, fstep: int, fw: int, fh: int) -> np.ndarray:
        """
        根据基础先验框, 在特征图上逐像素生成输入图像对应的先验框
        :param anchor_base: 预生成的基础先验框->[num_anchor_base, 4], 列坐标对应[x1, y1, x2, y2]
        :param fstep: 每个特征点映射回原图后在原图上的步进, 也就是空间缩放比例
        :param fw: 特征图像宽度
        :param fh: 特征图像高度
        :return: 生成的全图先验框->[num_anchor_base * fw * fh, 4], 列坐标对应[x1, y1, x2, y2]
        """
        # 特征图上每个点都会生成anchor, 第一个特征点对应原图上的先验框就是anchor_base, 由于特征图与原图间空间缩放, 相邻两特征点对应的anchor在原图上的步进为图像空间缩放尺度大小fstep
        # 因此由anchor_base和anchor间步进, 可以计算出输入图像的所有anchor

        # 计算出原图每一行上anchor与anchor_base的位置偏移值取值
        shift_x = np.arange(0, fw * fstep, fstep)
        # 计算出原图每一列上anchor与anchor_base的位置偏移值取值
        shift_y = np.arange(0, fh * fstep, fstep)
        # 用两方向偏移值生成网格, x和y方向偏移值维度均为[fh, fw], shift_x/shift_y相同位置的两个值表示当前anchor相对于anchor_base的坐标偏移值
        shift_x, shift_y = np.meshgrid(shift_x, shift_y)
        # 将shift_x/shift_y展开成一维并按列拼接在一起, 分别对应anchor的x1/y1/x2/y2的坐标偏移值, 构成偏移值矩阵
        shift = np.stack([shift_x.ravel(), shift_y.ravel(), shift_x.ravel(), shift_y.ravel()], axis=1)

        # anchor_base维度为[num_anchor_base, 4], 偏移值矩阵shift维度为[fw * fh, 4]
        # 将anchor_base每一行与shift中每一行元素相加即得到位置偏移后的所有anchor坐标, 但二者维度不同需进行扩展
        num_anchor_base = anchor_base.shape[0]
        num_points = shift.shape[0]
        # 将两者均扩展为[num_points, num_anchors, 4]
        anchor_base_extend = np.repeat(a=anchor_base[np.newaxis, :, :], repeats=num_points, axis=0)
        shift_extend = np.repeat(a=shift[:, np.newaxis, :], repeats=num_anchor_base, axis=1)
        # 获取最终anchors坐标, 并展开成二维向量, 维度为[num_anchors * num_points, 4] = [num_anchors * fw * fh, 4]
        anchors = anchor_base_extend + shift_extend
        anchors = np.reshape(a=anchors, newshape=(-1, 4))
        return anchors

    def forward(self, x: Tensor, im_size: tuple, scale: float = 1.0) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """
        前向处理, 获取rpn网络的回归器输出/分类器输出/矫正的roi区域/roi数据索引/全图先验框
        :param x: 由原图提取的输入特征图feature_map->[num, 512, fh, fw]
        :param im_size: 原始输入图像尺寸->[im_width, im_height]
        :param scale: 与min_size一起用于控制最小先验框尺寸
        :return: list->[回归器输出, 分类器输出, 建议框, 建议框对应的数据索引, 全图先验框]
        """
        num, chans, fh, fw = x.size()

        # 先将输入图像特征经过3*3网络, 进一步特征处理用于后续分类和回归
        x = self.conv(x)
        x = self.relu(x)

        # 将特征图送入分类网络, 计算特征图上每个像素的分类结果, 对应原输入图像所有先验框的分类结果, 维度为[num, num_anchor_base * 2, fh, fw]
        out_classifier = self.classifier(x)
        # 维度转换[num, num_anchor_base * 2, fh, fw]->[num, fh, fw, num_anchor_base * 2]->[num, fh * fw * num_anchor_base, 2]
        out_classifier = out_classifier.permute(0, 2, 3, 1).contiguous().view(num, -1, 2)

        # 将特征图送入回归网络, 计算特征图上每个像素的回归结果, 对应原输入图像所有先验框的回归结果, 维度为[num, num_anchor_base * 4, fh, fw]
        out_offsets = self.regressor(x)
        # 维度转换[num, num_anchor_base * 4, fh, fw]->[num, fh, fw, num_anchor_base * 4]->[num, fh * fw * num_anchor_base, 4]
        out_offsets = out_offsets.permute(0, 2, 3, 1).contiguous().view(num, -1, 4)

        # 将分类器输出转换为得分
        out_scores = self.softmax(out_classifier)
        # out_scores[:, :, 1]表示存在目标的概率
        out_scores = out_scores[:, :, 1].contiguous().view(num, -1)

        # 生成全图先验框, 每个特征点都会生成num_anchor_base个先验框, 故全图生成的先验框维度为[fh * fw * num_anchor_base, 4]
        anchors = self.generate_shifted_anchors(anchor_base=self.anchor_base, fstep=self.feature_stride, fw=fw, fh=fh)
        # 将anchors转换到和out_offsets的数据类型(float32)和设备类型(cuda/cpu)一致
        anchors = torch.tensor(data=anchors).type_as(out_offsets)

        # 获取batch数据的roi区域和对应的索引
        rois, rois_idx = [], []
        # 遍历batch中每个数据
        for i in range(num):
            # 按照数据索引获取当前数据对应的偏移值和得分, 生成固定数量的推荐区域, 维度为[fixed_num, 4]
            proposals = self.create_proposals(offsets=out_offsets[i], anchors=anchors, scores=out_scores[i], im_size=im_size, scale=scale)
            # 创建和推荐区域proposals数量相同的batch索引, 维度为[fixed_num]
            batch_idx = torch.tensor([i] * len(proposals))
            # 将推荐区域和索引进行维度扩展后放入列表中, 扩展后二者维度分别为[1 ,fixed_num, 4]和[[1 ,fixed_num]]
            rois.append(proposals.unsqueeze(0))
            rois_idx.append(batch_idx.unsqueeze(0))

        # 将rois列表中所有张量沿0维拼接在一起. 原rois列表长度为num, 其中每个张量均为[1, fixed_num, 4], 拼接后rois张量维度为[num, fixed_num, 4]
        rois = torch.cat(rois, dim=0).type_as(x)
        # 将rois索引拼接在一起, 拼接后维度为[num, fixed_num]
        rois_idx = torch.cat(rois_idx, dim=0).to(x.device)

        return out_offsets, out_classifier, rois, rois_idx, anchors


class ProposalTargetCreator:
    def __init__(self, num_samples=128, positive_iou_thresh=0.5, negative_iou_thresh=(0.5, 0.0), positive_rate=0.5):
        """
        在roi区域中选择一定数量的正负样本区域, 计算坐标偏移和分类标签, 用于后续分类和回归网络
        :param num_samples: 待保留的正负样本总数
        :param positive_iou_thresh: 正样本阈值
        :param negative_iou_thresh: 负样本阈值最大值和最小值
        :param positive_rate: 正样本比例
        """
        self.num_samples = num_samples
        self.positive_iou_thresh = positive_iou_thresh
        self.negative_iou_thresh = negative_iou_thresh
        self.positive_rate = positive_rate
        self.num_positive_per_image = int(num_samples * positive_rate)

        # 定义坐标偏移值归一化系数, 用于正负样本区域的offsets归一化
        self.offsets_normalize_params = np.array([[0, 0, 0, 0], [0.1, 0.1, 0.2, 0.2]], dtype="float32")

    def __call__(self, rois: np.ndarray, gt_boxes: np.ndarray, labels: np.ndarray):
        """
        根据推荐区域的iou结果选择一定数量的推荐区域作为正负样本, 并计算推荐区域与真值间的偏移值
        :param rois: 推荐区域, 维度为[m, 4]
        :param gt_boxes: 真值框, 维度为[n, 4]
        :param labels: 图像类别标签, 维度为[l, 1], 注意此处取值为[1, num_target_classes], 默认背景为0
        :return: 保留的正负样本区域/区域偏移值/区域标签
        """
        rois = np.concatenate((rois, gt_boxes), axis=0)

        # 计算iou结果
        iou = AnchorCreator.calc_IoU(anchors=rois, gt_boxes=gt_boxes)
        # 根据iou最大获取每个推荐框对应的真实框的idx和相应iou结果
        _, max_iou_idx_gt, max_iou_values = AnchorCreator.calc_max_iou_info(iou=iou)
        # 获取每个roi区域对应的真值框
        roi_gt_boxes = gt_boxes[max_iou_idx_gt]

        # 获取每个roi区域对应的真值框标签, 取值从1开始, 如果取值从0开始, 由于存在背景, 真值框标签需要额外加1
        roi_gt_labels = labels[max_iou_idx_gt]

        # 选取roi区域中的正样本序号, np.where()返回满足条件的元组, 元组第一个元素为行索引, 第二个元素为列索引
        positive_idx = np.where(max_iou_values >= self.positive_iou_thresh)[0]
        num_positive = min(self.num_positive_per_image, len(positive_idx))
        if len(positive_idx) > 1:
            positive_idx = np.random.choice(a=positive_idx, size=num_positive, replace=False)

        # 选取roi区域中的负样本序号
        negative_idx = np.where((max_iou_values < self.negative_iou_thresh[0]) & (max_iou_values >= self.negative_iou_thresh[1]))[0]
        num_negative = min(self.num_samples - num_positive, len(negative_idx))
        if len(negative_idx) > 1:
            negative_idx = np.random.choice(a=negative_idx, size=num_negative, replace=False)

        # 将正负样本索引整合在一起, 获得所有样本索引
        sample_idx = np.append(positive_idx, negative_idx)
        # 提取正负样本对应的真值标签, 此时无论正/负roi_gt_labels中都为对应iou最大的真值框标签, 下一步就需要对负样本标签赋值为0
        sample_labels = roi_gt_labels[sample_idx]
        # 对正负样本中的负样本标签赋值为0
        sample_labels[num_positive:] = 0
        # 提取样本对应的roi区域
        sample_rois = rois[sample_idx]

        # 计算选取的样本roi与真值的坐标偏移值
        # 根据样本索引, 获取样本对应的真值框
        sample_gt_boxes = roi_gt_boxes[sample_idx]
        # 计算推荐区域样本与真值的坐标偏移
        sample_offsets = AnchorCreator.calc_offsets_from_bboxes(anchors=sample_rois, target_boxes=sample_gt_boxes)
        # 对坐标偏移进行归一化
        sample_offsets = (sample_offsets - self.offsets_normalize_params[0]) / self.offsets_normalize_params[1]

        return sample_rois, sample_offsets, sample_labels


class ROIHead(nn.Module):
    def __init__(self, num_classes: int, pool_size: int, linear: nn.Module, spatial_scale: float = 1.0):
        """
        将ROI区域送入模型获得分类器输出和回归器输出
        :param num_classes: 样本类别数
        :param pool_size: roi池化目标尺寸
        :param linear: 线性模型
        :param spatial_scale: roi池化所使用的空间比例, 默认1.0, 若待处理的roi坐标为原图坐标, 则此处需要设置spatial_scale=目标特征图大小/原图大小
        """
        super(ROIHead, self).__init__()
        self.linear = linear

        # 对roi_pool结果进行回归预测
        self.regressor = nn.Linear(4096, num_classes * 4)
        self.classifier = nn.Linear(4096, num_classes)

        nn.init.kaiming_normal_(self.regressor.weight)
        nn.init.kaiming_normal_(self.classifier.weight)

        # 后续采用的roi坐标为特征图上坐标, 因此spatial_scale直接设置为1.0即可
        # 注意roi_pool要求roi坐标满足格式[x1, y1, x2, y2]
        self.roi_pool = RoIPool(output_size=(pool_size, pool_size), spatial_scale=spatial_scale)

    def forward(self, x: Tensor, rois: Tensor, rois_idx: Tensor, im_size: Tensor) -> Tuple[Tensor, Tensor]:
        """
        根据推荐框对特征图进行roi池化, 并将结果送入分类器和回归器, 得到相应结果
        :param x: 输入batch数据对应的全图图像特征, 维度为[num, 512, fh, fw]
        :param rois: 输入batch数据对应的rois区域, 维度为[num, num_samples, 4], 顺序为[y1, x1, y2, x2]
        :param rois_idx: 输入batch数据对应的rois区域索引, 维度为[num, num_samples]
        :param im_size: 原始输入图像尺寸, 维度为[im_height, im_width]
        :return: 分类模型和回归模型结果
        """
        num, chans, fh, fw = x.size()
        im_height, im_width = im_size

        # 将一个batch内数据的推荐区域展开堆叠在一起, 维度变为[num * num_samples, 4]
        rois = torch.flatten(input=rois, start_dim=0, end_dim=1)
        # 将一个batch内数据的索引展开堆叠在一起, 维度变为[num * num_samples]
        rois_idx = torch.flatten(input=rois_idx, start_dim=0, end_dim=1)

        # 计算原图roi区域映射到特征图后对应的边界框位置, 维度为[num * num_samples, 4]
        rois_on_features = torch.zeros_like(rois)
        # 计算[x1, x2]映射后的坐标
        rois_on_features[:, [0, 2]] = rois[:, [0, 2]] * (fw / im_width)
        # 计算[y1, y2]映射后的坐标
        rois_on_features[:, [1, 3]] = rois[:, [1, 3]] * (fh / im_height)

        # 将特征图上roi区域和对应的数据索引在列方向进行拼接, 得到[num * num_samples, 5]维张量, 用于后续roi_pool, 列对应[idx, x1, y1, x2, y2]
        fidx_and_rois = torch.cat([rois_idx.unsqueeze(1), rois_on_features], dim=1)

        # 根据数据idx和推荐框roi对输入图像特征图进行截取
        # 注意由于rois_on_features中roi坐标已经缩放到了特征图大小, 所以RoIPool池化时的spatial_scale需要设置为1.0
        # 注意此处roi_pool需要num * num_samples, 5]维, 根据第0列的idx取x中截取相应的特征进行池化
        pool_features = self.roi_pool(x, fidx_and_rois)

        # 上面获取的池化特征维度为[num * num_samples, chans, pool_size, pool_size], 将其展开为[num * num_samples, chans * pool_size * pool_size]以便送入分类和回归网络
        pool_features = pool_features.view(pool_features.size(0), -1)
        # 利用线性层进行特征进一步浓缩
        linear_features = self.linear(pool_features)

        # 将样本特征送入回归器, 得到各样本输出, 维度为[num * num_samples, 4 * num_classes]
        rois_out_regressor = self.regressor(linear_features)
        # 将样本特征送入回归器, 得到各样本输出, 维度为[num * num_samples, num_classes]
        rois_out_classifier = self.classifier(linear_features)

        # 维度变换, 获得当前batch中每个数据的所有回归结果, 维度为[num, num_samples, 4 * num_classes]
        rois_out_regressor = rois_out_regressor.view(num, -1, rois_out_regressor.size(1))
        # 维度变换, 获得当前batch中每个数据的所有分类结果, 维度为[num, num_samples, num_classes]
        rois_out_classifier = rois_out_classifier.view(num, -1, rois_out_classifier.size(1))

        return rois_out_regressor, rois_out_classifier


class FasterRCNN(nn.Module):
    def __init__(self, num_classes, train_flag=False, feature_stride=16, anchor_spatial_scales=(8, 16, 32), wh_ratios=(0.5, 1.0, 2.0), pretrained=False):
        """
        初始化Faster R-CNN
        :param num_classes: 最终分类类别数, 包含背景0和目标类别数
        :param train_flag: 是否为训练过程
        :param feature_stride: 特征步进, 实际就是特征提取器的空间缩放比例, 工程使用移除最后一个池化层的vgg16, 特征空间缩放比例为16
        :param anchor_spatial_scales: 待生成先验框与基本先验框的边长比值
        :param wh_ratios: 待生成先验框的宽高比
        :param pretrained: 特征提取器是否加载预训练参数
        """
        super(FasterRCNN, self).__init__()
        self.feature_stride = feature_stride
        self.extractor, linear = backbone(pretrained=pretrained)

        self.rpn = RegionProposalNet(in_channels=512, mid_channels=512, feature_stride=feature_stride, wh_ratios=wh_ratios,
                                     anchor_spatial_scales=anchor_spatial_scales, train_flag=train_flag)

        self.head = ROIHead(num_classes=num_classes, pool_size=POOL_SIZE, spatial_scale=1, linear=linear)

    def forward(self, x, scale: float = 1.0, mode: str = "forward"):
        """
        Faster R-CNN前向过程
        :param x: 输入
        :param scale: rpn结构中用于控制最小先验框尺寸
        :param mode: 处理流程控制字符串
        :return:
        """
        if mode == "forward":
            im_size = x.size()[-2:]
            # 提取输入图像特征
            im_features = self.extractor(x)
            # 获取建议框
            _, _, rois, rois_idx, _ = self.rpn(im_features, im_size, scale)
            # 根据图像特征和建议框计算偏移值回归结果和区域分类结果
            rois_out_regressor, rois_out_classifier = self.head(im_features, rois, rois_idx, im_size)
            return rois_out_regressor, rois_out_classifier, rois, rois_idx

        elif mode == "extractor":
            # 提取图像特征
            im_features = self.extractor(x)
            return im_features

        elif mode == "rpn":
            im_features, im_size = x
            # 获取建议框
            out_offsets, out_classes, rois, rois_idx, anchors = self.rpn(im_features, im_size, scale)
            return out_offsets, out_classes, rois, rois_idx, anchors

        elif mode == "head":
            im_features, rois, rois_idx, im_size = x
            # 获取分类和回归结果
            rois_out_regressor, rois_out_classifier = self.head(im_features, rois, rois_idx, im_size)
            return rois_out_regressor, rois_out_classifier

        else:
            raise TypeError("Invalid parameter of mode, which must be in ['forward', 'extractor', 'rpn', 'head']")


