"""
加载手机数据集图像，并进行中心裁剪预处理。
"""

import cv2
import numpy as np

def cropCenter(img, target_size):
    h, w = img.shape[:2]
    th, tw = target_size
    start_x = (w - tw) // 2
    start_y = (h - th) // 2
    return img[start_y:start_y+th, start_x:start_x+tw]

def loadPhone(pathData, imname):
    crop_l = (1344, 1008)
    
    file_path_right = f"{pathData}right_pd/{imname}/result_rightPd_center.png"
    file_path_left = f"{pathData}left_pd/{imname}/result_leftPd_center.png"
    file_path_guide = f"{pathData}scaled_images/{imname}/result_scaled_image_center.jpg"
    
    im_r = cv2.imread(file_path_right)
    im_l = cv2.imread(file_path_left)
    im_guide = cv2.imread(file_path_guide)
    
    if im_r is None or im_l is None or im_guide is None:
        raise ValueError("图像加载失败，请检查路径！")
    
    im_r = cv2.resize(cropCenter(im_r, crop_l), crop_l)
    im_l = cv2.resize(cropCenter(im_l, crop_l), crop_l)
    im_guide = cv2.resize(cropCenter(im_guide, crop_l), crop_l)
    
    # 将左右图像转换为灰度图
    if len(im_r.shape) == 3:
        im_r = cv2.cvtColor(im_r, cv2.COLOR_BGR2GRAY)
    if len(im_l.shape) == 3:
        im_l = cv2.cvtColor(im_l, cv2.COLOR_BGR2GRAY)
    if len(im_guide.shape) == 3:
        im_guide = cv2.cvtColor(im_guide, cv2.COLOR_BGR2GRAY)
    # 转换为 float32 类型并归一化到 [0,1] 范围
    im_r = im_r.astype('float32') / 255.0
    im_l = im_l.astype('float32') / 255.0
    
    # 此处假设 GT 深度及置信度图（若存在）为空，实际中请根据需要加载
    gt_depth = None
    conf_depth = None
    
    return im_l, im_r, im_guide, gt_depth, conf_depth