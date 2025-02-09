"""
针对手机数据的 CSGM 算法示例脚本。
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from CSGM.DP_Phone.utils.loadPhone import loadPhone

def main():
    pltFlag = 1
    dataFlag = 'phone'
    
    # 指定手机数据集图像名称与路径
    imname = 'sample_phone'
    pathData = r'C:\Users\local_admin\CSGM\DP_data\phone\\'
    
    # 加载手机数据（包括左图、右图、引导图、GT图、置信度图）
    im_l, im_r, im_guide, gt_depth, conf_depth = loadPhone(pathData, imname)
    
    # 后续调用针对手机的视差计算，此处仅展示图像加载结果
    plt.figure()
    plt.subplot(121)
    plt.imshow(im_l, cmap='gray')
    plt.title('Left Image')
    plt.subplot(122)
    plt.imshow(im_r, cmap='gray')
    plt.title('Right Image')
    plt.show()

if __name__ == '__main__':
    main()