import os
import torch
import numpy as np
import torch.nn as nn
from torch import Tensor
from typing import Tuple
import matplotlib.pyplot as plt
from Utils import GenDataSet
from torch.optim.lr_scheduler import StepLR
from Model import AnchorCreator, ProposalTargetCreator, FasterRCNN
from Config import *


def multitask_loss(out_offsets: Tensor, out_classifier: Tensor, gt_offsets: Tensor, gt_labels: Tensor, alpha: float = 1.0) -> Tuple[Tensor, Tensor, Tensor]:
    """
    计算多任务损失
    :param out_offsets: 回归模型边界框结果
    :param out_classifier: 分类模型边界框结果
    :param gt_offsets: 真实边界框
    :param gt_labels: 边界框标签
    :param alpha: 权重系数
    :return: 分类损失/正样本回归损失/总损失
    """
    # 分类损失计算式忽略标签为-1的样本
    cls_loss_func = nn.CrossEntropyLoss(ignore_index=-1)
    reg_loss_func = nn.SmoothL1Loss()

    # 计算分类损失
    loss_cls = cls_loss_func(out_classifier, gt_labels)

    # 选择正样本计算回归损失
    out_offsets_valid = out_offsets[gt_labels > 0]
    gt_offsets_valid = gt_offsets[gt_labels > 0]
    loss_reg = reg_loss_func(out_offsets_valid, gt_offsets_valid)

    # 总损失
    loss = loss_cls + alpha * loss_reg

    return loss_cls, loss_reg, loss


def train(data_set, network, num_epochs, optimizer, scheduler, device, train_rate: float = 0.8):
    """
    模型训练
    :param data_set: 训练数据集
    :param network: 网络结构
    :param num_epochs: 训练轮次
    :param optimizer: 优化器
    :param scheduler: 学习率调度器
    :param device: CPU/GPU
    :param train_rate: 训练集比例
    :return: None
    """
    os.makedirs('./model', exist_ok=True)
    network = network.to(device)
    best_loss = np.inf
    print("=" * 8 + "开始训练模型" + "=" * 8)
    # 计算训练batch数量
    batch_num = len(data_set)
    train_batch_num = round(batch_num * train_rate)
    # 记录训练过程中每一轮损失和准确率
    train_loss_all, val_loss_all, train_acc_all, val_acc_all = [], [], [], []
    anchor_creator = AnchorCreator()
    proposal_creator = ProposalTargetCreator()

    for epoch in range(num_epochs):
        # 记录train/val总损失
        num_train_loss = num_val_loss = train_loss = val_loss = 0.0

        for step, batch_data in enumerate(data_set):
            # 读取数据, 注意gt_boxes列坐标对应[x1, y1, x2, y2]
            ims, labels, gt_boxes = batch_data
            ims = ims.to(device)
            labels = labels.to(device)
            gt_boxes = gt_boxes.to(device)

            num, chans, im_height, im_width = ims.size()

            if step < train_batch_num:
                # 设置为训练模式
                network.train()
                # 获取输入图像全图特征, 维度为[num, 512, im_height/16, im_width/16]
                im_features = network.forward(x=ims, mode="extractor")

                # 利用rpn网络获取回归器输出/分类器输出/batch数据对应建议框/建议框对应数据索引/全图先验框
                rpn_offsets, rpn_classifier, rois, rois_idx, anchors = network.forward(x=[im_features, (im_height, im_width)], mode="rpn")
                np_anchors = anchors.cpu().numpy()

                # 记录rpn区域推荐网络的分类/回归损失, 以及最终的roi区域分类和回归损失
                rpn_cls_loss, rpn_reg_loss, roi_cls_loss, roi_reg_loss = 0, 0, 0, 0
                samples_rois, samples_indexes, samples_offsets, samples_labels = [], [], [], []

                # 遍历每一个数据的真值框/数据标签/rpn网络输出
                for i in range(num):
                    # 获取每张图像的真值框/标签/rpn生成的偏移值/rpn生成的分类输出
                    cur_gt_boxes = gt_boxes[i]
                    cur_labels = labels[i]
                    cur_rpn_offsets = rpn_offsets[i]
                    cur_rpn_classifier = rpn_classifier[i]
                    cur_rois = rois[i]

                    np_cur_gt_boxes = cur_gt_boxes.clone().detach().cpu().numpy()
                    np_cur_rois = cur_rois.clone().detach().cpu().numpy()
                    np_cur_labels = cur_labels.clone().detach().cpu().numpy()

                    # 根据当前图像真值框和先验框, 获取每个先验框标签以及图像经过rpn网络后应产生的偏移值作为真值
                    cur_gt_rpn_labels, cur_gt_rpn_offsets = anchor_creator(im_width=im_width, im_height=im_height, anchors=np_anchors, gt_boxes=np_cur_gt_boxes)

                    # 转换为张量后计算rpn网络的回归损失和分类损失
                    cur_gt_rpn_offsets = torch.tensor(cur_gt_rpn_offsets).type_as(rpn_offsets)
                    cur_gt_rpn_labels = torch.tensor(cur_gt_rpn_labels).long().to(rpn_offsets.device)
                    cur_rpn_cls_loss, cur_rpn_reg_loss, _ = multitask_loss(out_offsets=cur_rpn_offsets, out_classifier=cur_rpn_classifier,
                                                                           gt_offsets=cur_gt_rpn_offsets, gt_labels=cur_gt_rpn_labels)

                    rpn_cls_loss += cur_rpn_cls_loss
                    rpn_reg_loss += cur_rpn_reg_loss

                    # 在当前图像生成的roi建议框中中, 抽取一定数量的正负样本, 并计算出相应位置偏移, 用于后续回归和区域分类
                    sample_rois, sample_offsets, sample_labels = proposal_creator(rois=np_cur_rois, gt_boxes=np_cur_gt_boxes, labels=np_cur_labels)

                    # 将每个图像生成的样本信息存储起来用于后续回归和分类
                    # 抽取当前数据生成的推荐区域样本放入list中, list长度为num, 每个元素维度为[num_samples, 4]
                    samples_rois.append(torch.tensor(sample_rois).type_as(rpn_offsets))
                    # 将抽取的样本索引放入list中, list长度为num, 每个元素维度为[num_samples]
                    samples_indexes.append(torch.ones(len(sample_rois), device=rpn_offsets.device) * rois_idx[i][0])
                    # 将抽取的样本偏移值放入list中, list长度为num, 每个元素维度为[num_samples, 4]
                    samples_offsets.append(torch.tensor(sample_offsets).type_as(rpn_offsets))
                    # 将抽取的样本分类标签放入list中, list长度为num, 每个元素维度为[num_samples]
                    samples_labels.append(torch.tensor(sample_labels, device=rpn_offsets.device).long())

                # 整合当前batch数据抽取的推荐区域信息
                samples_rois = torch.stack(samples_rois, dim=0)
                samples_indexes = torch.stack(samples_indexes, dim=0)
                # 将图像特征和推荐区域送入模型, 进行roi池化并获得分类模型/回归模型输出
                roi_out_offsets, roi_out_classifier = network.forward(x=[im_features, samples_rois, samples_indexes, (im_height, im_width)], mode="head")

                # 遍历每帧图像的roi信息
                for i in range(num):
                    cur_num_samples = roi_out_offsets.size(1)
                    cur_roi_out_offsets = roi_out_offsets[i]
                    cur_roi_out_classifier = roi_out_classifier[i]
                    cur_roi_gt_offsets = samples_offsets[i]
                    cur_roi_gt_labels = samples_labels[i]

                    # 将当前数据的roi区域由[cur_num_samples, num_classes * 4] -> [cur_num_samples, num_classes, 4]
                    cur_roi_out_offsets = cur_roi_out_offsets.view(cur_num_samples, -1, 4)
                    # 根据roi对应的样本标签, 选择与其类别对应真实框的offsets
                    cur_roi_offsets = cur_roi_out_offsets[torch.arange(0, cur_num_samples), cur_roi_gt_labels]

                    # 计算分类网络和回归网络的损失值
                    cur_roi_cls_loss, cur_roi_reg_loss, _ = multitask_loss(out_offsets=cur_roi_offsets, out_classifier=cur_roi_out_classifier,
                                                                           gt_offsets=cur_roi_gt_offsets, gt_labels=cur_roi_gt_labels)

                    roi_cls_loss += cur_roi_cls_loss
                    roi_reg_loss += cur_roi_reg_loss

                # 计算整体loss, 反向传播
                batch_loss = (rpn_cls_loss + rpn_reg_loss + roi_cls_loss + roi_reg_loss) / num
                optimizer.zero_grad()
                batch_loss.backward()
                optimizer.step()

                # 记录每轮训练数据量和总loss
                train_loss += batch_loss.item() * num
                num_train_loss += num

            else:
                # 设置为验证模式
                network.eval()

                with torch.no_grad():
                    # 获取输入图像全图特征, 维度为[num, 512, im_height/16, im_width/16]
                    im_features = network.forward(x=ims, mode="extractor")

                    # 利用rpn网络获取回归器输出/分类器输出/batch数据对应建议框/建议框对应数据索引/全图先验框
                    rpn_offsets, rpn_classifier, rois, rois_idx, anchors = network.forward(x=[im_features, (im_height, im_width)], mode="rpn")
                    np_anchors = anchors.cpu().numpy()

                    # 记录rpn区域网络的分类/回归损失, 以及roi区域分类和回归损失
                    rpn_cls_loss, rpn_reg_loss, roi_cls_loss, roi_reg_loss = 0, 0, 0, 0
                    samples_rois, samples_indexes, samples_offsets, samples_labels = [], [], [], []

                    # 遍历每一个数据的真值框/数据标签/rpn网络输出
                    for i in range(num):
                        # 获取每张图像的真值框/标签/rpn生成的偏移值/rpn生成的分类输出
                        cur_gt_boxes = gt_boxes[i]
                        cur_labels = labels[i]
                        cur_rpn_offsets = rpn_offsets[i]
                        cur_rpn_classifier = rpn_classifier[i]
                        cur_rois = rois[i]

                        np_cur_gt_boxes = cur_gt_boxes.clone().detach().cpu().numpy()
                        np_cur_rois = cur_rois.clone().detach().cpu().numpy()
                        np_cur_labels = cur_labels.clone().detach().cpu().numpy()

                        # 根据当前图像真值框和先验框, 获取每个先验框标签以及图像经过rpn网络后应产生的偏移值作为真值
                        cur_gt_rpn_labels, cur_gt_rpn_offsets = anchor_creator(im_width=im_width, im_height=im_height,
                                                                               anchors=np_anchors, gt_boxes=np_cur_gt_boxes)

                        # 转换为张量后计算rpn网络的回归损失和分类损失
                        cur_gt_rpn_offsets = torch.tensor(cur_gt_rpn_offsets).type_as(rpn_offsets)
                        cur_gt_rpn_labels = torch.tensor(cur_gt_rpn_labels).long().to(rpn_offsets.device)
                        cur_rpn_cls_loss, cur_rpn_reg_loss, _ = multitask_loss(out_offsets=cur_rpn_offsets, out_classifier=cur_rpn_classifier,
                                                                               gt_offsets=cur_gt_rpn_offsets, gt_labels=cur_gt_rpn_labels)

                        rpn_cls_loss += cur_rpn_cls_loss
                        rpn_reg_loss += cur_rpn_reg_loss

                        # 在当前图像生成的roi建议框中中, 抽取一定数量的正负样本, 并计算出相应位置偏移, 用于后续回归和区域分类
                        sample_rois, sample_offsets, sample_labels = proposal_creator(rois=np_cur_rois, gt_boxes=np_cur_gt_boxes, labels=np_cur_labels)

                        # 将每个图像生成的样本信息存储起来用于后续回归和分类
                        # 抽取当前数据生成的推荐区域样本放入list中, list长度为num, 每个元素维度为[num_samples, 4]
                        samples_rois.append(torch.tensor(sample_rois).type_as(rpn_offsets))
                        # 将抽取的样本索引放入list中, list长度为num, 每个元素维度为[num_samples]
                        samples_indexes.append(torch.ones(len(sample_rois), device=rpn_offsets.device) * rois_idx[i][0])
                        # 将抽取的样本偏移值放入list中, list长度为num, 每个元素维度为[num_samples, 4]
                        samples_offsets.append(torch.tensor(sample_offsets).type_as(rpn_offsets))
                        # 将抽取的样本分类标签放入list中, list长度为num, 每个元素维度为[num_samples]
                        samples_labels.append(torch.tensor(sample_labels, device=rpn_offsets.device).long())

                    # 整合当前batch数据抽取的推荐区域信息
                    samples_rois = torch.stack(samples_rois, dim=0)
                    samples_indexes = torch.stack(samples_indexes, dim=0)
                    # 将图像特征和推荐区域送入模型, 进行roi池化并获得分类模型/回归模型输出
                    roi_out_offsets, roi_out_classifier = network.forward(x=[im_features, samples_rois, samples_indexes, (im_height, im_width)], mode="head")

                    for i in range(num):
                        cur_num_samples = roi_out_offsets.size(1)
                        cur_roi_out_offsets = roi_out_offsets[i]
                        cur_roi_out_classifier = roi_out_classifier[i]
                        cur_roi_gt_offsets = samples_offsets[i]
                        cur_roi_gt_labels = samples_labels[i]

                        cur_roi_out_offsets = cur_roi_out_offsets.view(cur_num_samples, -1, 4)
                        # 根据roi对应的样本标签, 选择与其类别对应真实框的offsets
                        cur_roi_offsets = cur_roi_out_offsets[torch.arange(0, cur_num_samples), cur_roi_gt_labels]

                        # 计算分类网络和回归网络的损失值
                        cur_roi_cls_loss, cur_roi_reg_loss, _ = multitask_loss(out_offsets=cur_roi_offsets, out_classifier=cur_roi_out_classifier,
                                                                               gt_offsets=cur_roi_gt_offsets, gt_labels=cur_roi_gt_labels)

                        roi_cls_loss += cur_roi_cls_loss
                        roi_reg_loss += cur_roi_reg_loss

                    # 计算整体loss, 反向传播
                    batch_loss = (rpn_cls_loss + rpn_reg_loss + roi_cls_loss + roi_reg_loss) / num

                    # 记录每轮训练数据量和总loss
                    val_loss += batch_loss.item() * num
                    num_val_loss += num

        scheduler.step()
        # 记录loss和acc变化曲线
        train_loss_all.append(train_loss / num_train_loss)
        val_loss_all.append(val_loss / num_val_loss)
        print("Epoch:[{:0>3}|{}]  train_loss:{:.3f}  val_loss:{:.3f}".format(epoch + 1, num_epochs, train_loss_all[-1], val_loss_all[-1]))

        # 保存模型
        if val_loss_all[-1] < best_loss:
            best_loss = val_loss_all[-1]
            save_path = os.path.join("./model", "model_" + str(epoch + 1) + ".pth")
            torch.save(network, save_path)

    # 绘制训练曲线
    fig_path = os.path.join("./model/",  "train_curve.png")
    plt.plot(range(num_epochs), train_loss_all, "r-", label="train")
    plt.plot(range(num_epochs), val_loss_all, "b-", label="val")
    plt.title("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_path)
    plt.close()

    return None


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = FasterRCNN(num_classes=CLASSES, train_flag=True, feature_stride=FEATURE_STRIDE, anchor_spatial_scales=ANCHOR_SPATIAL_SCALES,
                       wh_ratios=ANCHOR_WH_RATIOS, pretrained=False)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=LR)
    scheduler = StepLR(optimizer, step_size=STEP, gamma=GAMMA)

    model_root = "./model"
    os.makedirs(model_root, exist_ok=True)

    # 在生成的ss数据上进行预训练
    train_root = "./data/source"
    train_set = GenDataSet(root=train_root, im_width=IM_SIZE, im_height=IM_SIZE, batch_size=BATCH_SIZE, shuffle=True)

    train(data_set=train_set, network=model, num_epochs=EPOCHS, optimizer=optimizer, scheduler=scheduler, device=device, train_rate=0.8)
