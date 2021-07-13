"""
@File : my_dataset.py
@Author : CodeCat
@Time : 2021/7/12 下午5:02
"""
import numpy as np
import cv2
from PIL import Image
import math
import os
import torch
import torchvision.transforms as transforms
from torch.utils import data

from utils.data_aug import adjust_height, rotate_img, crop_img, rotate_vertices
from utils.label_generation_utils import shrink_poly, find_min_rect_angle, get_boundary, rotate_all_pixels


def extract_vertices(lines):
    """
    从文本行中提取顶点信息
    :param
        lines: <list> 字符串信息列表
    :return:
        vertices：<numpy.ndarray, (n, 8)> 文本区域的顶点
        labels: <numpy.ndarray, (n, )> 1表示有效文本，0表示无效文本
    """
    labels = []
    vertices = []
    for line in lines:
        # 377,117,463,117,465,130,378,130,Genaxis Theatre -> [377,117,463,117,465,130,378,130]
        vertices.append(list(map(int, line.rstrip('\n').lstrip('\ufeff').split(',')[:8])))
        label = 0 if '###' in line else 1
        labels.append(label)
    return np.array(vertices), np.array(labels)


def get_score_geo(img, vertices, labels, scale, length):
    """
    生成score gt 和 RBOX geometry gt
    :param img: PIL Image
    :param vertices: <numpy.ndarray, (n, 8)> 文本区域顶点信息
    :param labels: <numpy.ndarray, (n, )> 1表示有效文本，0表示无效文本
    :param scale: feature map / image  1/4
    :param length: image length
    :return:
        score gt
        geo gt
        ignored
    """
    score_map = np.zeros((int(img.height * scale), int(img.width * scale), 1), np.float32)
    geo_map = np.zeros((int(img.height * scale), int(img.width * scale), 5), np.float32)
    ignored_map = np.zeros((int(img.height * scale), int(img.width * scale), 1), np.float32)

    index = np.arange(0, length, int(1 / scale))
    index_y, index_x = np.meshgrid(index, index)
    ignored_polys = []
    polys = []

    for i, vertice in enumerate(vertices):
        if labels[i] == 0:
            ignored_poly = (scale * vertice.reshape((4, 2))).astype(np.int32)
            ignored_polys.append(ignored_poly)
            continue

        poly = (scale * shrink_poly(vertice).reshape((4, 2))).astype(np.int32)
        polys.append(poly)
        temp_mask = np.zeros(score_map.shape[:-1], np.float32)
        cv2.fillPoly(temp_mask, [poly], 1)

        theta = find_min_rect_angle(vertice)
        rotate_mat = np.array([[math.cos(theta), -math.sin(theta)],
                               [math.sin(theta), math.cos(theta)]])

        rotated_vertice = rotate_vertices(vertice, theta)
        x_min, x_max, y_min, y_max = get_boundary(rotated_vertice)
        rotated_x, rotated_y = rotate_all_pixels(rotate_mat, vertice[0], vertice[1], length)

        d1 = rotated_y - y_min
        d1[d1 < 0] = 0
        d2 = y_max - rotated_y
        d2[d2 < 0] = 0
        d3 = rotated_x - x_min
        d3[d3 < 0] = 0
        d4 = x_max - rotated_x
        d4[d4 < 0] = 0
        # 像素点到矩形顶部的距离
        geo_map[:, :, 0] += d1[index_x, index_y] * temp_mask
        # 像素点到矩形底部的距离
        geo_map[:, :, 1] += d2[index_x, index_y] * temp_mask
        # 像素点到矩形左侧的距离
        geo_map[:, :, 2] += d3[index_x, index_y] * temp_mask
        # 像素点到矩形右侧的距离
        geo_map[:, :, 3] += d4[index_x, index_y] * temp_mask
        # 旋转角度
        geo_map[:, :, 4] += theta * temp_mask

    cv2.fillPoly(ignored_map, ignored_polys, 1)
    cv2.fillPoly(score_map, polys, 1)
    return torch.Tensor(score_map).permute(2, 0, 1), torch.Tensor(geo_map).permute(2, 0, 1), torch.Tensor(ignored_map).permute(2, 0, 1)


class Custom_dataset(data.Dataset):
    def __init__(self, img_path, gt_path, scale=0.25, length=512):
        super(Custom_dataset, self).__init__()
        self.img_files = [os.path.join(img_path, img_file) for img_file in sorted(os.listdir(img_path))]
        self.gt_files = [os.path.join(gt_path, gt_file) for gt_file in sorted(os.listdir(gt_path))]
        self.scale = scale
        self.length = length

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, item):
        with open(self.gt_files[item], 'r') as f:
            lines = f.readlines()

        vertices, labels = extract_vertices(lines)

        img = Image.open(self.img_files[item])
        img, vertices = adjust_height(img, vertices)
        img, vertices = rotate_img(img, vertices)
        img, vertices = crop_img(img, vertices, labels, self.length)
        transform = transforms.Compose(
            [
                transforms.ColorJitter(0.5, 0.5, 0.5, 0.25),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
            ]
        )
        score_map, geo_map, ignored_map = get_score_geo(img, vertices, labels, self.scale, self.length)
        return transform(img), score_map, geo_map, ignored_map