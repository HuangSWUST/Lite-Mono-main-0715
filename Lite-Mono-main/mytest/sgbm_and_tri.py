import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
import math

cam_params = {}

cam_params["intr_1"] = np.array(
    [[0.25 * 1.457572438721727e+003, 0.000000000000000e+00, 0.25 * 1.212945694211622e+003 - 30],
     [0.000000000000000e+00, 0.25 * 1.457522226502963e+003, 0.25 * 1.007320588489210e+003 - 30],
     [0.000000000000000e+00, 0.000000000000000e+00, 1.000000000000000e+00]],
    dtype=np.float64)

cam_params["intr_0"] = np.array(
    [[0.25 * 1.460868570835972e+003, 0.000000000000000e+00, 0.25 * 1.215024068023046e+003 - 30],
     [0.000000000000000e+00, 0.25 * 1.460791367088000e+003, 0.25 * 1.011107202932225e+003 - 30],
     [0.000000000000000e+00, 0.000000000000000e+00, 1.000000000000000e+00]],
    dtype=np.float64)

cam_params["dis_1"] = np.array([[-3.202059184174811e-002, 5.484810041960487e-002, -2.376797821038721e-004,
                                 1.957176410761953e-003, 0.000000000000000e+00]], dtype=np.float64)

cam_params["dis_0"] = np.array([[-3.010775294366006e-002, 5.207379006686103e-002, 1.275242537506779e-003,
                                 6.132156106101780e-005, 0.000000000000000e+00]], dtype=np.float64)

tmp_R = np.array([[9.9854401420328975e-01, 2.9595497349581539e-02, -4.5099426109552407e-02],
                  [-3.0994227401413992e-02, 9.9904990776121294e-01, -3.0637226867026929e-02],
                  [4.4149853528290657e-02, 3.1990441368370412e-02, 9.9851259486021859e-01]],
                 dtype=np.float64)

cam_params["R"] = tmp_R.transpose()

tmp_T = np.array([[9.9960400252946990e-01], [1.1883594323257373e-02], [2.5507220801641667e-02]],
                 dtype=np.float64)
cam_params["T"] = -np.dot(cam_params["R"], tmp_T)

imagesize = [544, 448]
RL, RR, PL, PR, Q, roiL, roiR = cv2.stereoRectify(cam_params["intr_0"], 0, cam_params["intr_1"],
                                                  0, imagesize, cam_params["R"], cam_params["T"])
cam_params["R0"] = RL
cam_params["R1"] = RR

cam_params["P0"] = PL
cam_params["P1"] = PR


def disp_cau(img0, img1):
    image_width = img0.shape[1]
    image_height = img0.shape[0]

    # cv2.imshow("img0", img0)
    # cv2.imshow("img1", img1)
    # # 窗口等待
    # cv2.waitKey(0)

    blockSize = 5
    img_channels = 1
    minDisparity = 0
    numDisparities = 96

    zeros = np.zeros((image_height, numDisparities), dtype=img0.dtype)
    img_resize_l = np.concatenate([zeros, img0], axis=1)
    img_resize_r = np.concatenate([zeros, img1], axis=1)

    stereo = cv2.StereoSGBM_create(minDisparity=minDisparity,
                                   numDisparities=numDisparities,  # 16倍数
                                   blockSize=blockSize,
                                   P1=2 * img_channels * blockSize * blockSize,
                                   P2=64 * img_channels * blockSize * blockSize,
                                   disp12MaxDiff=-1,
                                   preFilterCap=15,  # sobel处理的映射滤波器大小
                                   uniquenessRatio=1,
                                   speckleWindowSize=-20,  # 连通域窗口尺寸
                                   speckleRange=1,  # 连通域内最大视差变化，隐式x16
                                   # mode=cv2.STEREO_SGBM_MODE_HH
                                   )
    # stereo = cv2.StereoBM_create(numDisparities=64, blockSize=15)
    # 计算视差
    imgDisparity16S = stereo.compute(img_resize_l, img_resize_r)[:, numDisparities:]
    print(imgDisparity16S.dtype)

    image_width = imgDisparity16S.shape[1]
    image_height = imgDisparity16S.shape[0]

    mat = np.zeros((image_height, image_width))

    for i in range(0, image_height):

        for j in range(0, image_width):
            temp = imgDisparity16S[i, j] / 16
            if (temp <= minDisparity) | (temp > numDisparities):
                continue
            mat[i, j] = temp

    # imgDisparity8 = imgDisparity16S.astype(np.uint8)
    # print(imgDisparity8.dtype)
    # 将视差图归一化
    # min_val = disparity_16.min()
    # max_val = disparity_16.max()
    # disparity = np.uint8(6400 * (disparity_16 - min_val) / (max_val - min_val))

    # 视差图可视化
    disp_img = mat.astype(np.float)
    kernel = np.ones((7, 7), np.float)

    # 图像腐蚀处理
    erosion = cv2.erode(disp_img, kernel)
    # 图像膨胀处理
    result = cv2.dilate(erosion, kernel)
    result = cv2.dilate(result, kernel)

    # # 显示窗口
    # cv2.imshow("erosion", erosion)
    # cv2.imshow("result", result)
    # cv2.imshow("disp_img", disp_img)
    # # 窗口等待
    # cv2.waitKey(0)

    return result


def unrectify(point, cam_params, flag):
    camP = cam_params['P1']
    camI = cam_params["intr_1"]
    camR = cam_params["R1"]
    if flag:
        camP = cam_params['P0']
        camI = cam_params["intr_0"]
        camR = cam_params["R0"]

    xyw = np.ones([3, 1])

    xyw[0] = point[0] - camP[0, 2] / camP[0, 0]
    xyw[1] = point[1] - camP[1, 2] / camP[1, 1]
    xyw = np.dot(camR, xyw)
    xyw[0] /= xyw[2]
    xyw[1] /= xyw[2]
    # tmp = xyw[0] * camI[0, 0] + camI[0, 2]
    xyw_new = np.array([xyw[0] * camI[0, 0] + camI[0, 2],
                        xyw[1] * camI[1, 1] + camI[1, 2]])

    return xyw_new


# 像素点三角化
def my_tri(p, q, R, T):
    Af = np.ones((4, 3), dtype=np.float)
    Bf = np.ones((4, 1), dtype=np.float)

    A = np.ones(9, dtype=np.float)
    b = np.ones(3, dtype=np.float)
    x = np.ones(3, dtype=np.float)

    # prepare Af
    Af[0] = -1.0
    Af[1] = 0.0
    Af[2] = p[0]
    Af[3] = 0.0
    Af[4] = -1.0
    Af[5] = p[1]
    Af[6] = q[0] * R(2, 0) - R(0, 0)
    Af[7] = q[0] * R(2, 1) - R(0, 1)
    Af[8] = q[0] * R(2, 2) - R(0, 2)
    Af[9] = q[1] * R(2, 0) - R(1, 0)
    Af[10] = q[1] * R(2, 1) - R(1, 1)
    Af[11] = q[1] * R(2, 2) - R(1, 2)

    # prepare Bf
    Bf[0] = 0.0
    Bf[1] = 0.0
    Bf[2] = T(0, 0) - T(2, 0) * q[0]
    Bf[3] = T(1, 0) - T(2, 0) * q[1]

    #  Compute A= Af'Af
    A[0] = Af[0] * Af[0] + Af[3] * Af[3] + Af[6] * Af[6] + Af[9] * Af[9]
    A[1] = Af[0] * Af[1] + Af[3] * Af[4] + Af[10] * Af[9] + Af[6] * Af[7]
    A[2] = Af[0] * Af[2] + Af[3] * Af[5] + Af[11] * Af[9] + Af[6] * Af[8]

    A[3] = A[1]
    A[4] = Af[1] * Af[1] + Af[10] * Af[10] + Af[4] * Af[4] + Af[7] * Af[7]
    A[5] = Af[10] * Af[11] + Af[1] * Af[2] + Af[4] * Af[5] + Af[7] * Af[8]

    A[6] = A[2]
    A[7] = A[5]
    A[8] = Af[11] * Af[11] + Af[2] * Af[2] + Af[5] * Af[5] + Af[8] * Af[8]

    #  Compute b = Af'*Bf
    b[0] = Af[0] * Bf[0] + Af[3] * Bf[1] + Af[6] * Bf[2] + Af[9] * Bf[3]
    b[1] = Af[1] * Bf[0] + Af[10] * Bf[3] + Af[4] * Bf[1] + Af[7] * Bf[2]
    b[2] = Af[2] * Bf[0] + Af[11] * Bf[3] + Af[5] * Bf[1] + Af[8] * Bf[2]

    A.reshape(3, 3)
    cv2.solve(A.reshape(3, 3), b, x, cv2.DECOMP_LU)

    return x


def triangulate(imgl, imgr, disp, cam_params):
    min_angle = 30
    cam_distance = 2.5
    height = imgl.shape[0]
    width = imgl.shape[1]
    point_mesh = []
    for xl in range(0, imgl.shape[0]):
        for yl in range(0, imgl.shape[0]):
            if disp[xl, yl] > 0:
                xr = xl + disp[xl, yl]
                yr = yl

                # 相机坐标系
                pi = unrectify([xl, yl], cam_params, True)
                qi = unrectify([xr, yr], cam_params, False)

                if pi[0] < 1 | pi[0] >= width - 1 | pi[1] < 1 | pi[1] >= height - 1 \
                        | qi[0] < 1 | qi[0] >= width - 1 | qi[1] < 1 | qi[1] >= height - 1:
                    continue

                p = [pi[0] - cam_params['intr_0'][0, 2] / cam_params['intr_0'][0, 0],
                     pi[1] - cam_params['intr_0'][1, 2] / cam_params['intr_0'][1, 1]]

                q = [qi[0] - cam_params['intr_1'][0, 2] / cam_params['intr_1'][0, 0],
                     qi[1] - cam_params['intr_1'][1, 2] / cam_params['intr_1'][1, 1]]
                # 角度约束
                if min_angle > 0:
                    d1 = cv2.normalize(np.array([p[0], p[1], 1]))
                    d2 = np.array([q[0], q[1], 1])
                    d2 = cv2.normalize(cam_params['R'] * d2 + cam_params['T'])
                    ang = abs(math.acos(d1.ddot(d2)) * 57.29577951)
                    if ang < min_angle:
                        continue
                # 三角化
                p3d = my_tri(p, q, cam_params['R'], cam_params['T'])
                # 距离约束
                ptdistance = cv2.norm(p3d)
                if ptdistance < cam_distance / 10:
                    continue
                if ptdistance > cam_distance * 200:
                    continue
                R = imgr[xr, yr]

                point_mesh.append(np.array([p3d[0], p3d[1], p3d[2], R, R, R]))
    return point_mesh


imgl = cv2.imread("D:/DataSet/undist_resize/train/input1/0.png", cv2.IMREAD_GRAYSCALE)
imgr = cv2.imread("D:/DataSet/undist_resize/train/input2/0.png", cv2.IMREAD_GRAYSCALE)

# 左图视差
disp = disp_cau(imgl, imgr)
point_mesh = triangulate(imgl, imgr, disp, cam_params)
