"""
封装手机数据集的 CSGM 算法实现，
支持多层金字塔处理，完全复现 MATLAB 版本的处理流程。
"""

import cv2
import numpy as np
from CSGM.DP_Phone.utils.currPhoneLvlCSGM import currPhoneLvlCSGM

def build_pyramid(image, levels):
    """
    构建图像金字塔，返回列表，列表第一个元素为最低分辨率，最后一个为原图。
    使用 cv2.pyrDown 实现。
    """
    pyr = []
    current = image.copy()
    for i in range(levels):
        if i > 0:
            current = cv2.pyrDown(current)
        pyr.append(current)
    pyr = pyr[::-1]
    return pyr

def wrapperPhoneCSGM(im_L, im_R, im_guide, params):
    """
    根据参数进行多层处理。如果 levels > 1，则先在最低层计算粗略视差，
    再逐层上采样并利用先验信息进行细化。

    返回:
      dispar_map_pyr: 列表，每个元素为一个金字塔层的视差图
      sum_parab: 最后层拟合参数
      conf_score_no_suprress: 置信度
    """
    levels = params.get('levels', 1)
    dispar_map_pyr = []
    conf_score_no_suprress = None

    if levels > 1:
        pyr_L = build_pyramid(im_L, levels)
        pyr_R = build_pyramid(im_R, levels)
        pyr_guide = build_pyramid(im_guide, levels)
        
        params['levels'] = levels
        # 第一级：最低分辨率
        disp_coarse, sum_parab, conf_score_no_suprress = currPhoneLvlCSGM(
            pyr_L[0], pyr_R[0], pyr_guide[0], params, None, None)
        dispar_map_pyr.append(disp_coarse)
        
        # 逐层细化
        for lvl in range(1, levels):
            # 当前层尺寸与上一层尺寸的比例（通常为2倍）
            scale = pyr_L[lvl].shape[0] / pyr_L[lvl-1].shape[0]
            
            # --- 对上一层视差进行上采样，并乘以尺度因子 ---
            up_disp = cv2.resize(dispar_map_pyr[-1],
                                 (pyr_L[lvl].shape[1], pyr_L[lvl].shape[0]),
                                 interpolation=cv2.INTER_LINEAR)
            up_disp = up_disp * scale

            # --- 对低层拟合参数进行上采样和尺度矫正 ---
            if sum_parab is not None and 'a' in sum_parab:
                sum_parab['a'] = sum_parab['a'] * ((1/scale)**2)
                sum_parab['b'] = sum_parab['b'] * (1/scale)
                sum_parab['a'] = cv2.resize(sum_parab['a'],
                                            (pyr_L[lvl].shape[1], pyr_L[lvl].shape[0]),
                                            interpolation=cv2.INTER_LINEAR)
                sum_parab['b'] = cv2.resize(sum_parab['b'],
                                            (pyr_L[lvl].shape[1], pyr_L[lvl].shape[0]),
                                            interpolation=cv2.INTER_LINEAR)
                sum_parab['c'] = cv2.resize(sum_parab['c'],
                                            (pyr_L[lvl].shape[1], pyr_L[lvl].shape[0]),
                                            interpolation=cv2.INTER_LINEAR)


            disp_refined, sum_parab, conf_score_no_suprress = currPhoneLvlCSGM(
                pyr_L[lvl], pyr_R[lvl], pyr_guide[lvl], params, sum_parab, up_disp)
            dispar_map_pyr.append(disp_refined)
    else:
        disp, sum_parab, conf_score_no_suprress = currPhoneLvlCSGM(im_L, im_R, im_guide, params, None, None)
        dispar_map_pyr.append(disp)
    
    return dispar_map_pyr, sum_parab, conf_score_no_suprress