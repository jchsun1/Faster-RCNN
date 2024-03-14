import os
import torch
import numpy as np
from torch import Tensor
import torch.nn.functional as F
from torchvision.ops import nms
from Model import FasterRCNN, ProposalCreator
import matplotlib.pyplot as plt
from matplotlib import patches
from Utils import GenDataSet
from skimage import io
from Config import *


def draw_box(img: np.ndarray, boxes: np.ndarray = None, save_name: str = None):
    """
    在图像上绘制边界框
    :param img: 输入图像
    :param boxes: bbox坐标, 列分别为[x, y, w, h, score, label]
    :param save_name: 保存bbox图像名称, None-不保存
    :return: None
    """
    plt.imshow(img)
    axis = plt.gca()
    if boxes is not None:
        for box in boxes:
            x, y, w, h = box[:4].astype("int")
            score = box[4]
            rect = patches.Rectangle((x, y), w, h, linewidth=1, edgecolor='r', facecolor='none')
            axis.add_patch(rect)
            axis.text(x, y - 10, "Score: {:.2f}".format(score), fontsize=12, color='blue')
    if save_name is not None:
        os.makedirs("./predict", exist_ok=True)
        plt.savefig("./predict/" + save_name + ".jpg")
    plt.show()
    return None


def predict(network: FasterRCNN, im: np.ndarray, device: torch.device, im_width: int, im_height: int, num_classes: int,
            offsets_norm_params: Tensor, nms_thresh: float = 0.3, confidence_thresh: float = 0.5, save_name: str = None):
    """
    模型预测
    :param network: Faster R-CNN模型结构
    :param im: 原始输入图像矩阵
    :param device: CPU/GPU
    :param im_width: 输入模型的图像宽度
    :param im_height: 输入模型的图像高度
    :param num_classes: 目标类别数
    :param offsets_norm_params: 偏移值归一化参数
    :param nms_thresh: 非极大值抑制阈值
    :param confidence_thresh: 目标置信度阈值
    :param save_name: 保存文件名
    :return: None
    """
    # 测试模式
    network.eval()

    src_height, src_width = im.shape[:2]
    # 数据归一化和缩放
    im_norm = (im / 255.0).astype("float32")
    im_rsz = GenDataSet.resize(im=im_norm, im_width=im_width, im_height=im_height, gt_boxes=None)
    # 将矩阵转换为张量
    im_tensor = torch.tensor(np.transpose(im_rsz, (2, 0, 1))).unsqueeze(0).to(device)
    with torch.no_grad():
        # 获取Faster R-CNN网络的输出, 包括回归器输出/分类器输出/推荐区域
        # 维度分别为[num_ims, num_rois, num_classes * 4]/[num_ims, num_rois, num_classes]/[num_ims, num_rois, 4]
        rois_out_regressor, rois_out_classifier, rois, _ = network.forward(x=im_tensor, mode="forward")

        # 获取当前图像数量/推荐区域数量
        num_ims, num_rois, _ = rois_out_regressor.size()

        # 记录预测的边界框信息
        out_bboxes = []
        # 遍历处理每张图片, 此处实际只有一张图
        cur_rois_offsets = rois_out_regressor[0]
        cur_rois = rois[0]
        cur_rois_classifier = rois_out_classifier[0]

        # 将偏移值进行维度变换[num_rois, num_classes * 4] -> [num_rois, num_classes, 4]
        cur_rois_offsets = cur_rois_offsets.view(-1, num_classes, 4)
        # 对roi区域进行维度变换[num_rois, 4] -> [num_rois, 1, 4] -> [num_rois, num_classes, 4]
        cur_rois = cur_rois.view(-1, 1, 4).expand_as(cur_rois_offsets)

        # 将偏移值和roi区域展开成相同大小的二维张量, 方便对roi进行位置矫正
        # 将偏移值展开, 维度[num_rois, num_classes, 4] -> [num_rois * num_classes, 4]
        cur_rois_offsets = cur_rois_offsets.view(-1, 4)
        # 将和roi区域展开, 维度[num_rois, num_classes, 4] -> [num_rois * num_classes, 4]
        cur_rois = cur_rois.contiguous().view(-1, 4)

        # 对回归结果进行修正
        # 注意Faster R-CNN网络输出的样本偏移值是经过均值方差修正的, 此处需要将其还原
        offsets_norm_params = offsets_norm_params.type_as(cur_rois_offsets)
        # ProposalTargetCreator中计算方式: sample_offsets = (sample_offsets - self.offsets_normalize_params[0]) / self.offsets_normalize_params[1]
        cur_rois_offsets = cur_rois_offsets * offsets_norm_params[1] + offsets_norm_params[0]

        # 利用偏移值对推荐区域位置进行矫正, 获得预测框位置, 维度[num_rois * num_classes, 4]
        cur_target_boxes = ProposalCreator.calc_bboxes_from_offsets(offsets=cur_rois_offsets, anchors=cur_rois)
        # 展开成[num_rois, num_classes, 4]方便与类别一一对应
        cur_target_boxes = cur_target_boxes.view(-1, num_classes, 4)

        # 获取分类得分
        cur_roi_scores = F.softmax(cur_rois_classifier, dim=-1)

        # 根据目标得分, 获取最可能的分类结果
        max_prob_labels = torch.argmax(input=cur_roi_scores[:, 1:], dim=-1) + 1
        max_prob_scores = cur_roi_scores[torch.arange(0, cur_roi_scores.size(0)), max_prob_labels]
        max_prob_boxes = cur_target_boxes[torch.arange(0, cur_target_boxes.size(0)), max_prob_labels]

        # 选取得分大于阈值的
        is_valid_scores = max_prob_scores > confidence_thresh

        if sum(is_valid_scores) > 0:
            valid_boxes = max_prob_boxes[is_valid_scores]
            valid_scores = max_prob_scores[is_valid_scores]
            keep = nms(boxes=valid_boxes, scores=valid_scores, iou_threshold=nms_thresh)

            # 获取保留的目标框, 维度为[num_keep, 4]
            keep_boxes = valid_boxes[keep]
            # 获取保留目标框的得分, 并将维度扩展为[num_keep, 1]
            keep_scores = valid_scores[keep][:, None]

            # 将预测框/标签/得分堆叠在一起
            cls_predict = torch.cat([keep_boxes, keep_scores], dim=1).cpu().numpy()

            # 预测结果添加进cur_out_bboxes里
            out_bboxes.extend(cls_predict)

        if len(out_bboxes) > 0:
            out_bboxes = np.array(out_bboxes)
            # 计算原始输入图像和模型输入图像之间的空间缩放比例
            map_scale = np.array([src_width, src_height, src_width, src_height]) / np.array([im_width, im_height, im_width, im_height])
            # 将预测框从模型输入图像映射到原始输入图像
            out_bboxes[:, :4] = out_bboxes[:, :4] * map_scale
            # 对预测框坐标进行限制
            out_bboxes[:, [0, 2]] = np.clip(a=out_bboxes[:, [0, 2]], a_min=0, a_max=src_width - 1)
            out_bboxes[:, [1, 3]] = np.clip(a=out_bboxes[:, [1, 3]], a_min=0, a_max=src_height - 1)
            # 将预测框[x1, y1, x2, y2, score, label]转换为[x1, y1, w, h, score, label]
            out_bboxes[:, [2, 3]] = out_bboxes[:, [2, 3]] - out_bboxes[:, [0, 1]] + 1

        if len(out_bboxes) == 0:
            out_bboxes = None
        draw_box(img=im, boxes=out_bboxes, save_name=save_name)


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_path = "./model/model_180.pth"
    model = torch.load(model_path, map_location=device)

    # 偏移值归一化参数, 需保证和训练阶段一致
    offsets_normalize_params = torch.Tensor([[0, 0, 0, 0], [0.1, 0.1, 0.2, 0.2]])

    test_root = "./data/source/17flowers"
    for roots, dirs, files in os.walk(test_root):
        for file in files:
            if not file.endswith(".jpg"):
                continue
            im_name = file.split(".")[0]
            im_path = os.path.join(roots, file)
            im = io.imread(im_path)
            predict(network=model, im=im, device=device, im_width=IM_SIZE, im_height=IM_SIZE, num_classes=CLASSES, offsets_norm_params=offsets_normalize_params,
                    nms_thresh=NMS_THRESH, confidence_thresh=CONFIDENCE_THRESH, save_name=im_name)
