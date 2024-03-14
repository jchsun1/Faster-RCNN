# 模型输入图像大小
IM_SIZE = 480

# roi-pool大小
POOL_SIZE = 7

# 模型输入图像上的anchor位置步进, 由于使用VGG16前4个2*2池化, 故步进为2^4=16
FEATURE_STRIDE = 16

# 先验框大小和宽高比控制参数
ANCHOR_SPATIAL_SCALES = (8, 16, 32)
ANCHOR_WH_RATIOS = (0.5, 1.0, 2.0)

# 训练过程参数
CLASSES = 3
BATCH_SIZE = 3
EPOCHS = 200
LR = 0.0001
STEP = 20
GAMMA = 0.6

# 预测过程参数
NMS_THRESH = 0.2
CONFIDENCE_THRESH = 0.9
