"""
@File : detect.py
@Author : CodeCat
@Time : 2021/7/13 下午3:33
"""
import torch
from torchvision import transforms
from PIL import Image, ImageDraw
from model.east_vgg16 import EAST
import numpy as np
import lanms
import math
import time


def resize_img(img):
    """将图像resize至能整除32"""
    w, h = img.size
    resize_w, resize_h = w, h

    resize_w = resize_w if resize_w % 32 == 0 else int(resize_w / 32) * 32
    resize_h = resize_h if resize_h % 32 == 0 else int(resize_h / 32) * 32

    img = img.resize((resize_w, resize_h), Image.BILINEAR)
    ratio_w = resize_w / w
    ratio_h = resize_h / h

    return img, ratio_w, ratio_h


def load_pil(img):
    """将PIL Image转换为Tensor"""
    t = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(.5, .5, .5), std=(.5, .5, .5))
    ])
    return t(img).unsqueeze(0)


def get_rotate_mat(theta):
    return np.array([[math.cos(theta), -math.sin(theta)],
                     [math.sin(theta), math.cos(theta)]])


def is_valid_poly(res, score_shape, scale):
    """检查文本区域是否在图像内"""
    cnt = 0
    for i in range(res.shape[1]):
        if res[0, i] < 0 or res[0, i] >= score_shape[1] * scale or res[1, i] < 0 or res[1, i] >= score_shape[0] * scale:
            cnt += 1
    return True if cnt <= 1 else False


def restore_polys(valid_pos, valid_geo, score_shape, scale=4):
    """从特征图给定的有效位置恢复文本区域"""
    polys = []
    index = []
    valid_pos *= scale
    d = valid_geo[:4, :]
    angel = valid_geo[4, :]

    for i in range(valid_pos.shape[0]):
        x = valid_pos[i, 0]
        y = valid_pos[i, 1]
        y_min = y - d[0, i]
        y_max = y + d[1, i]
        x_min = x - d[2, i]
        x_max = x + d[3, i]
        rotate_mat = get_rotate_mat(-angel[i])

        temp_x = np.array([[x_min, x_max, x_max, x_min]]) - x
        temp_y = np.array([[y_min, y_min, y_max, y_max]]) - y
        coordidates = np.concatenate((temp_x, temp_y), 0)
        res = np.dot(rotate_mat, coordidates)
        res[0, :] += x
        res[1, :] += y

        if is_valid_poly(res, score_shape, scale):
            index.append(i)
            polys.append([res[0, 0], res[1, 0], res[0, 1], res[1, 1],
                          res[0, 2], res[1, 2], res[0, 3], res[1, 3]])
    return np.array(polys), index


def get_boxes(score, geo, score_thresh=0.9, nms_thresh=0.2):
    """从特征图中获得boxes"""
    score = score[0, :, :]
    xy_text = np.argwhere(score > score_thresh)
    if xy_text.size == 0:
        return None

    # 按行排序
    xy_text = xy_text[np.argsort(xy_text[:, 0])]
    valid_pos = xy_text[:, ::-1].copy()
    valid_geo = geo[:, xy_text[:, 0], xy_text[:, 1]]
    polys_restored, index = restore_polys(valid_pos, valid_geo, score.shape)
    if polys_restored.size == 0:
        return None

    boxes = np.zeros((polys_restored.shape[0], 9), dtype=np.float32)
    boxes[:, :8] = polys_restored
    boxes[:, 8] = score[xy_text[index, 0], xy_text[index, 1]]
    boxes = lanms.merge_quadrangle_n9(boxes.astype('float32'), nms_thresh)
    return boxes


def adjust_ratio(boxes, ratio_w, ratio_h):
    if boxes is None or boxes.size == 0:
        return None
    boxes[:, [0, 2, 4, 6]] /= ratio_w
    boxes[:, [1, 3, 5, 7]] /= ratio_h
    return np.around(boxes)


def detect(img, model, device):
    """使用训练好的模型检查图像的文本区域"""
    img, ratio_w, ratio_h = resize_img(img)
    with torch.no_grad():
        score, geo = model(load_pil(img).to(device))
    boxes = get_boxes(score.squeeze(0).cpu().numpy(), geo.squeeze(0).cpu().numpy())
    return adjust_ratio(boxes, ratio_w, ratio_h)


def plot_boxes(img, boxes):
    """在图像上绘制文本区域"""
    if boxes is None:
        return img

    draw = ImageDraw.Draw(img)
    for box in boxes:
        draw.polygon([box[0], box[1], box[2], box[3], box[4], box[5], box[6], box[7]], outline=(0, 255, 0))

    return img


if __name__ == '__main__':
    img_path = './ICDAR_2015/test_img/img_91.jpg'
    model_path = './east_vgg16.pth'
    res_img = './detect_test/res.bmp'
    res_box = './detect_test/res.txt'
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = EAST().to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    img = Image.open(img_path)

    start = time.time()
    boxes = detect(img, model, device)
    end = time.time()
    print("检测一张图像耗时为：{:.4f}s".format(end - start))
    print(boxes)
    with open(res_box, 'w') as f:
        f.write(str(boxes))
    plot_img = plot_boxes(img, boxes)
    plot_img.save(res_img)