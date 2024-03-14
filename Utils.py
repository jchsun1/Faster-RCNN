import os
import torch
import numpy as np
import random
import skimage.io as io
from typing import Union
import pandas as pd
import cv2 as cv


class GenDataSet:
    def __init__(self, root: str, im_width: int, im_height: int, batch_size: int, shuffle: bool = False,
                 prob_vertical_flip: float = 0.5, prob_horizontal_flip: float = 0.5):
        """
        初始化GenDataSet
        :param root: 数据路径
        :param im_width: 目标图片宽度
        :param im_height: 目标图片高度
        :param batch_size: 批数据大小
        :param shuffle: 是否随机打乱批数据
        :param prob_vertical_flip: 随机垂直翻转概率
        :param prob_horizontal_flip: 随机水平翻转概率
        """
        self.root = root
        csv_path = os.path.join(root, "gt_loc.csv")
        self.flist = pd.read_csv(csv_path, header=0, index_col=None)

        self.im_width, self.im_height = (im_width, im_height)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_batch = self.calc_num_batch()
        self.prob_vertical_flip = prob_vertical_flip
        self.prob_horizontal_flip = prob_horizontal_flip

    def calc_num_batch(self) -> int:
        """
        计算batch数量
        :return: 批数据数量
        """
        total = self.flist.shape[0]
        if total % self.batch_size == 0:
            num_batch = total // self.batch_size
        else:
            num_batch = total // self.batch_size + 1
        return num_batch

    @staticmethod
    def normalize(im: np.ndarray) -> np.ndarray:
        """
        将图像数据归一化
        :param im: 输入图像->uint8
        :return: 归一化图像->float32
        """
        if im.dtype != np.uint8:
            raise TypeError("uint8 img is required.")
        else:
            im = im / 255.0
        im = im.astype("float32")
        return im

    @staticmethod
    def resize(im: np.ndarray, im_width: int, im_height: int, gt_boxes: Union[np.ndarray, None]):
        """
        对图像进行缩放
        :param im: 输入图像
        :param im_width: 目标图像宽度
        :param im_height: 目标图像高度
        :param gt_boxes: 真实边界框->[n, 4], 列分别对应[x1, y1, w, h]
        :return: 图像和两种边界框经过缩放后的结果
        """
        rows, cols = im.shape[:2]
        # 图像缩放
        im = cv.resize(src=im, dsize=(im_width, im_height), interpolation=cv.INTER_CUBIC)

        if gt_boxes is not None:
            # 计算缩放过程中(x, y, w, h)尺度缩放比例
            scale_ratio = np.array([im_width / cols, im_height / rows, im_width / cols, im_height / rows])
            # 边界框也等比例缩放
            gt_boxes = gt_boxes * scale_ratio
            return im, gt_boxes
        return im

    def random_horizontal_flip(self, im: np.ndarray, gt_boxes: np.ndarray):
        """
        随机水平翻转图像
        :param im: 输入图像
        :param gt_boxes: 边界框真值, 列坐标对应[x1, y1, w, h]
        :return: 翻转后图像和边界框结果
        """
        if random.uniform(0, 1) < self.prob_horizontal_flip:
            rows, cols = im.shape[:2]
            # 左右翻转图像
            im = np.fliplr(im)
            # 边界框位置重新计算
            gt_boxes[:, 0] = cols - 1 - gt_boxes[:, 0] - gt_boxes[:, 2]
        else:
            pass
        return im, gt_boxes

    def random_vertical_flip(self, im: np.ndarray, gt_boxes: np.ndarray):
        """
        随机垂直翻转图像
        :param im: 输入图像
        :param gt_boxes: 边界框真值, 列坐标对应[x1, y1, w, h]
        :return: 翻转后图像和边界框结果
        """
        if random.uniform(0, 1) < self.prob_vertical_flip:
            rows, cols = im.shape[:2]
            # 上下翻转图像
            im = np.flipud(im)
            # 重新计算边界框位置
            gt_boxes[:, 1] = rows - 1 - gt_boxes[:, 1] - gt_boxes[:, 3]
        else:
            pass
        return im, gt_boxes

    def get_fdata(self):
        """
        数据集准备
        :return: 数据列表, 注意生成的真值框列坐标对应[x1, y1, x2, y2]
        """
        fdata = []
        flist = self.flist
        if self.shuffle:
            flist = flist.sample(frac=1).reset_index(drop=True)

        for num in range(self.num_batch):
            # 按照batch大小读取数据
            cur_flist = flist[num * self.batch_size: (num + 1) * self.batch_size]
            # 记录当前batch的图像/标签/真实边界框
            cur_ims, cur_labels, cur_gt_boxes = [], [], []
            for idx in range(cur_flist.shape[0]):
                name = cur_flist.iat[idx, 0]
                label = cur_flist.iat[idx, 1]
                gt_boxes = cur_flist.iloc[idx, 2: 6].values.astype("float32")
                gt_boxes = gt_boxes[np.newaxis, :]

                # 读取图像
                im_path = os.path.join(self.root, name)
                img = io.imread(im_path)

                # 数据归一化
                img = self.normalize(im=img)
                # 随机翻转数据增强
                img, gt_boxes = self.random_horizontal_flip(im=img, gt_boxes=gt_boxes)
                img, gt_boxes = self.random_vertical_flip(im=img, gt_boxes=gt_boxes)
                # 将图像缩放到统一大小
                img, gt_boxes = self.resize(im=img, im_width=self.im_width, im_height=self.im_height, gt_boxes=gt_boxes)

                # 转换为tensor
                im_tensor = torch.tensor(np.transpose(img, (2, 0, 1)))
                # 将gt_boxes由[x1, y1, w, h]转换为[x1, y1, x2, y2]
                gt_boxes[:, 2] = gt_boxes[:, 0] + gt_boxes[:, 2] - 1
                gt_boxes[:, 3] = gt_boxes[:, 1] + gt_boxes[:, 3] - 1
                gt_boxes_tensor = torch.tensor(gt_boxes)

                cur_ims.append(im_tensor)
                cur_labels.append([label])
                cur_gt_boxes.append(gt_boxes_tensor)

            # 每个batch数据放一起方便后续训练调用
            cur_ims = torch.stack(cur_ims)
            cur_labels = torch.tensor(cur_labels)
            cur_gt_boxes = torch.stack(cur_gt_boxes)
            fdata.append([cur_ims, cur_labels, cur_gt_boxes])
        return fdata

    def __len__(self):
        # 以batch数量定义数据集大小
        return self.num_batch

    def __iter__(self):
        self.fdata = self.get_fdata()
        self.index = 0
        return self

    def __next__(self):
        if self.index >= self.num_batch:
            raise StopIteration
        # 生成当前batch数据
        value = self.fdata[self.index]
        self.index += 1
        return value
