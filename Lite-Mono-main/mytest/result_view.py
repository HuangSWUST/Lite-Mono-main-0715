import os
from glob import glob

import cv2
import numpy as np


def mask_process(mask):
    ret, binary = cv2.threshold(mask, 0.9, 1, cv2.THRESH_BINARY)
    # disp_img = mat.astype(np.float)
    kernel = np.ones((7, 7), np.float)

    # # 图像腐蚀处理
    # erosion = cv2.erode(binary, kernel)
    # # 图像膨胀处理
    # result1 = cv2.dilate(erosion, kernel)
    # result2 = cv2.dilate(result1, kernel)
    result1 = cv2.dilate(binary, kernel)
    result2 = cv2.erode(result1, kernel)
    # result3 = cv2.erode(result2, kernel)
    erosion = cv2.erode(result2, kernel)

    # cv2.imshow('erosion', erosion)
    # cv2.imshow('result1', result1)
    # cv2.imshow('result2', result2)
    # # cv2.imshow('result2', result3)
    # cv2.imshow('mask', mask)
    # cv2.imshow('binary', binary)
    # cv2.waitKey(0)
    return erosion


test_path = "D:/Data/project_py/Lite-Mono-main/Lite-Mono-main/test/input1"
test_image_path = os.path.join(test_path, "*.png")
maskl_path = os.path.join(test_path, "maskl*.npy")
displ_path = os.path.join(test_path, "displ*.npy")

image_names = glob(test_image_path)
maskl_names = glob(maskl_path)
displ_names = glob(displ_path)

for i in range(0, len(image_names)):
    imgl = cv2.imread(image_names[i], cv2.IMREAD_GRAYSCALE)
    # imgr = cv2.imread(image_names[i], cv2.IMREAD_GRAYSCALE)
    height = imgl.shape[0]
    width = imgl.shape[1]

    # disp_l_path = 'test/input1/displ_resized0.npy'
    # mask_l_path = 'test/input1/maskl_resized0.npy'

    disp_l = np.load(displ_names[i])
    mask_l = np.load(maskl_names[i])
    mask_l = mask_l / 0.3

    disp_img = disp_l.squeeze(0).squeeze(0)
    mask_img = mask_l.squeeze(0).squeeze(0)

    # 掩膜处理
    binary = mask_process(mask_img)

    disp_img_mask = disp_img * binary
    dtype = imgl.dtype
    img_mask = (imgl * binary).astype(dtype)

    cv2.imshow('img_mask', img_mask)
    cv2.imshow('disp_img_mask', disp_img_mask)
    cv2.imshow('disp_img', disp_img)
    cv2.imshow('mask_img', mask_img)
    cv2.imshow('binary', binary)
    cv2.waitKey(0)
