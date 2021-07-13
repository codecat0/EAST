## EAST
### 1. 项目描述
- 通过PyTorch简单实现了[EAST: An Efficient and Accurate Scene Text Detector](https://openaccess.thecvf.com/content_cvpr_2017/papers/Zhou_EAST_An_Efficient_CVPR_2017_paper.pdf)
- 参考了[songdejia/EAST](https://github.com/songdejia/EAST) 和 [SakuraRiven/EAST](https://github.com/SakuraRiven/EAST)
### 2. 代码结构
```
|--model
|   |--vgg.py : VGG网络结构
|   |--resnet.py ： ResNet网络结构
|   |--east_vgg16.py : VGG16 + EAST 
|   |--east_resnet50 : ResNet50 + EAST
|--utils
|   |--data_aug.py : 图像增强相关
|   |--label_generation_utils.py : 生成训练时所需的label
|   |--loss.py : 损失计算
|--my_dataset.py : 数据处理
|--train.py : 模型训练
|--detect.py : 模型预测
```
### 3. 数据集
- [ICDAR_2015](https://rrc.cvc.uab.es/?ch=4&com=downloads)

### 4. 环境配置
```
torch==1.8.1
numpy==1.20.2
Pillow==8.2.0
Shapely==1.7.1
lanms==1.0.2
opencv-python==4.5.2.54
```

### 5. 模型效果展示
`检测一张图像耗时为：0.4136s`
![](./res.bmp)
由于训练次数以及参数设置的原因，可以发现，效果不是特别好。