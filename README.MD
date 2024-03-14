# Faster RCNN
Object detection using Faster RCNN on the 17flowers dataset.

# 博客
算法原理及思路请参考本人博客：https://www.cnblogs.com/Haitangr/p/17742226.html

# 数据集
17flowers据集, 官网下载地址：http://www.robots.ox.ac.uk/~vgg/data/flowers/17/

# 文件说明
data						  数据目录，data/source用于存放原始数据，包括用于训练的2flowers和用于测试的17flowers\
doc							  文档目录，用于存放参考论文\
Predict.py				预测代码，用于最终的目标检测\
Model.py			    定义模型各个功能模块\
Train.py					模型训练代码\
Utils.py					用于数据集加载\
Config.py         目标检测训练和预测中的参数设置\
requirements.txt	环境依赖

# 程序运行流程
(1) 运行Train.py，程序会在训练数据集上进行Faster RCNN分类和回归多任务训练，并生成./model/路径保存模型\
(2) 运行Predict.py，对数据集进行预测，生成./predict/目录保存预测结果
