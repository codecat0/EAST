"""
@File : data_aug.py
@Author : CodeCat
@Time : 2021/7/12 下午5:53
"""
from shapely.geometry import Polygon
import numpy as np
from PIL import Image
import math


def adjust_height(img, vertices, ratio=0.2):
    """
    调整图像的高度用于数据增强
    :param img: PIL Image
    :param vertices: <numpy.ndarray, (n, 8)>  文本区域的顶点
    :param ratio: 高度调整的比率
    :return:
        img : 调整后的PIL Image
        new_vertices : 调整后的文本区域顶点
    """
    ratio_h = 1 + ratio * (np.random.rand() * 2 - 1)
    old_h = img.height
    new_h = int(old_h * ratio_h)
    img = img.resize((img.width, new_h), Image.BILINEAR)

    new_vertices = vertices.copy()
    if vertices.size > 0:
        new_vertices[:, [1, 3, 5, 7]] = vertices[:, [1, 3, 5, 7]] * (new_h / old_h)
    return img, new_vertices


def rotate_vertices(vertices, theta, anchor=None):
    """
    围绕anchor旋转vertices
    :param vertices: <numpy.ndarray, (8, )> 文本区域顶点
    :param theta: 旋转角度
    :param anchor: 在旋转过程中保持不变的位置
    :return:
        旋转后的文本区域顶点
    """
    v = vertices.reshape((4, 2)).T
    if anchor is None:
        anchor = v[:, :1]
    # 旋转矩阵
    rotate_mat = np.array([[math.cos(theta), -math.sin(theta)],
                           [math.sin(theta), math.cos(theta)]])
    res = np.dot(rotate_mat, v - anchor)
    return (res + anchor).T.reshape(-1)


def rotate_img(img, vertices, angle_range=10):
    """
    旋转图像用于数据增强，旋转角度为[-10, 10]
    :param img: PIL Image
    :param vertices: <numpy.ndarray, (n, 8)> 文本区域顶点
    :param angle_range: 旋转角度
    :return:
        img : 旋转后的PIL Image
        new_vertices : 旋转后的文本区域顶点
    """
    center_x = (img.width - 1) / 2
    center_y = (img.height - 1) / 2
    angle = angle_range * (np.random.rand() * 2 - 1)
    img = img.rotate(angle, Image.BILINEAR)
    new_vertices = np.zeros(vertices.shape)
    for i, vertice in enumerate(vertices):
        # 由于原始图像为逆时针旋转，所以角度为负
        new_vertices[i, :] = rotate_vertices(vertices=vertice,
                                             theta=-angle / 180 * math.pi,
                                             anchor=np.array([[center_x], [center_y]]))
    return img, new_vertices


def is_cross_text(start_loc, length, vertices):
    """
    检查裁剪的图像是否穿过文本区域
    :param start_loc: 左上角坐标
    :param length: 裁剪后图像的长度
    :param vertices: 文本区域顶点
    :return:
        如果裁剪的图像穿过文本区域，则返回Ture
    """
    if vertices.size == 0:
        return False
    start_w, start_h = start_loc
    a = np.array([start_w, start_h, start_w + length, start_h,
                  start_w + length, start_h + length, start_w, start_h + length]).reshape((4, 2))
    p1 = Polygon(a).convex_hull
    for vertice in vertices:
        p2 = Polygon(vertice.reshape((4, 2))).convex_hull
        # 计算p1和p2的相交面积
        inter = p1.intersection(p2).area
        if 0.01 <= inter / p2.area <= 0.99:
            return True
    return False


def crop_img(img, vertices, labels, length):
    """
    裁剪图像到指定大小以满足batch和数据增强
    :param img: PIL Image
    :param vertices: <numpy.ndarray, (n, 8)> 文本区域顶点
    :param labels: <numpy.ndarray, (n, )> 1表示有效文本，0表示无效文本
    :param length: 裁剪后图像的长度
    :return:
        region : 裁剪后的图像区域
        new_vertices : 裁剪后图像中文本区域顶点信息
    """
    h, w = img.height, img.width
    # 确保最短的边>=length
    if h >= w and w < length:
        img = img.resize((length, int(h * length / w)), Image.BILINEAR)
    elif h < w and h < length:
        img = img.resize((int(w * length / h), length), Image.BILINEAR)

    ratio_w = img.width / w
    ratio_h = img.height / h

    new_vertices = np.zeros(vertices.shape)
    if vertices.size > 0:
        new_vertices[:, [0, 2, 4, 6]] = vertices[:, [0, 2, 4, 6]] * ratio_w
        new_vertices[:, [1, 3, 5, 7]] = vertices[:, [1, 3, 5, 7]] * ratio_h

    # 寻找最佳的裁剪起始点
    start_w = 0
    start_h = 0
    remain_h = img.height - length
    remain_w = img.width - length
    flag = True
    cnt = 0
    while flag and cnt < 1000:
        cnt += 1
        start_w = int(np.random.rand() * remain_w)
        start_h = int(np.random.rand() * remain_h)
        flag = is_cross_text(start_loc=[start_w, start_h],
                             length=length,
                             vertices=new_vertices[labels==1, :])

    box = (start_w, start_h, start_w + length, start_h + length)
    region = img.crop(box)
    if new_vertices.size == 0:
        return region, new_vertices

    new_vertices[:, [0, 2, 4, 6]] -= start_w
    new_vertices[:, [1, 3, 5, 7]] -= start_h
    return region, new_vertices

