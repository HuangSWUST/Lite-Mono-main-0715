import cv2
import os
import numpy as np
from glob import glob


def read_path(file_pathname):
    #遍历该目录下的所有图片文件

    source_filename1 = glob("F:\wass\AA2014\input\cam0\*.png")
    source_filename2 = glob("F:\wass\AA2014\plane_fit\*.png")

    # source_filename1 = glob("D:/Data/project_py/da/test/input1/*.png")
    # source_filename2 = glob("D:/Data/project_py/da/test/input2/*.png")

    cam_0 = np.array([[1.457572438721727e+003,   0.000000000000000e+00,    1.212945694211622e+003],
                      [0.000000000000000e+00,    1.457522226502963e+003,    1.007320588489210e+003],
                      [0.000000000000000e+00,    0.000000000000000e+00,    1.000000000000000e+00]], dtype=np.float64)

    dis_0 = np.array([[-3.202059184174811e-002, 5.484810041960487e-002, -2.376797821038721e-004, 1.957176410761953e-003, 0.000000000000000e+00]], dtype=np.float64)

    cam_1 = np.array([[1.460868570835972e+003,   0.000000000000000e+00,    1.215024068023046e+003],
                      [0.000000000000000e+00,    1.460791367088000e+003 ,    1.011107202932225e+003],
                      [0.000000000000000e+00,    0.000000000000000e+00,    1.000000000000000e+00]], dtype=np.float64)

    dis_1 = np.array([[-3.010775294366006e-002, 5.207379006686103e-002, 1.275242537506779e-003, 6.132156106101780e-005, 0.000000000000000e+00]], dtype=np.float64)

    # R = np.array([[0.99854401420328975, 0.029595497349581539, -0.045099426109552407],
    #               [-0.030994227401413992, 0.99904990776121294, -0.030637226867026929],
    #               [0.044149853528290657, 0.031990441368370412, 0.99851259486021859]], dtype=np.float64)

    R = np.array([[9.9854401420328975e-01, 2.9595497349581539e-02, -4.5099426109552407e-02],
                  [-3.0994227401413992e-02, 9.9904990776121294e-01, -3.0637226867026929e-02],
                  [4.4149853528290657e-02, 3.1990441368370412e-02, 9.9851259486021859e-01]], dtype=np.float64)

    # t = np.array([[2499.0100063236746], [29.70898580814343], [63.76805200410417]], dtype=np.float64)
    t = np.array([[9.9960400252946990e-01], [1.1883594323257373e-02], [2.5507220801641667e-02]], dtype=np.float64)

    imagesize = (2456, 2058)


    # new_cam0 =  getOptimalNewCameraMatrix(cam0, dis_0,
    # imagesize,
    # double
    # alpha, Size
    # newImgSize = Size(), CV_OUT
    # Rect * validPixROI = 0,
    #                      bool
    # centerPrincipalPoint = false);


    # RL, RR, PL, PR, Q, roiL, roiR = cv2.stereoRectify(cam_0, dis_0, cam_1, dis_1, imagesize, R, t, flags=0, alpha=1)
    # RL, RR, PL, PR, Q, roiL, roiR = cv2.stereoRectify(cam_0, dis_0, cam_1, dis_1, imagesize, R, t, 0, 1, (0, 0))
    RL, RR, PL, PR, Q, roiL, roiR = cv2.stereoRectify(cam_0, dis_0, cam_1, dis_1, imagesize, R, t)

    Left_Stereo_Map = cv2.initUndistortRectifyMap(cam_0, dis_0, RL, PL, imagesize, cv2.CV_32FC1)
    Rigt_Stereo_Map = cv2.initUndistortRectifyMap(cam_1, dis_1, RR, PR, imagesize, cv2.CV_32FC1)


    for j in range(0, 2500):
        # imgl = cv2.imread(source_filename1[j], cv2.IMREAD_GRAYSCALE)
        imgr = cv2.imread(source_filename2[j], cv2.IMREAD_GRAYSCALE)

        # imgl = cv2.undistort(imgl, cam_0, dis_0)
        # imgr = cv2.undistort(imgr, cam_1, dis_1)

        # imagesize = imgl.shape[::-1]

        # imagesize = np.array([[2058], [2456]], dtype=np.float32)

        # Left_rectified = cv2.remap(imgl, Left_Stereo_Map[0], Left_Stereo_Map[1], cv2.INTER_CUBIC, cv2.BORDER_REPLICATE)
        # Rigt_rectified = cv2.remap(imgr, Rigt_Stereo_Map[0], Rigt_Stereo_Map[1], cv2.INTER_CUBIC,
        #                            cv2.BORDER_CONSTANT, 0)
        Rigt_rectified = cv2.remap(imgr, Rigt_Stereo_Map[0], Rigt_Stereo_Map[1], cv2.INTER_CUBIC, cv2.BORDER_REPLICATE)


        # Left_rectified = cv2.resize(Left_rectified, dsize=None, fx=0.25, fy=0.25, interpolation=cv2.INTER_LINEAR)
        Rigt_rectified = cv2.resize(Rigt_rectified, dsize=None, fx=0.25, fy=0.25, interpolation=cv2.INTER_LINEAR)


        #
        # Left_rectified = Left_rectified[30:478, 30:574]
        Rigt_rectified = Rigt_rectified[30:478, 30:574]


        cv2.imwrite("F:/wass/AA2014/resize_planefit/" + str(j) + ".png", Rigt_rectified);
        # cv2.imwrite("F:/wass/AA2014/undist_resize/input2/" + str(j) + ".png", Left_rectified);


        # cv2.imwrite("D:/DataSet/wass2k/input1/" + str(j) + ".png", Left_rectified);
        # cv2.imwrite("D:/DataSet/wass2k/input2/" + str(j) + ".png", Rigt_rectified);

        # cv2.imwrite("D:/Data/project_py/da/test/wassroi/input1/" + str(j) + ".png", Left_rectified);
        # cv2.imwrite("D:/Data/project_py/da/test/wassroi/input2/" + str(j) + ".png", Rigt_rectified);

        # image_merge = np.concatenate([Left_rectified, Rigt_rectified], axis=1)
        # cv2.namedWindow("image_merge", cv2.WINDOW_FREERATIO)
        # cv2.imshow("image_merge", image_merge)
        # cv2.imwrite("D:/DataSet/image_mergeroi.png", image_merge);
        # cv2.waitKey(0)

#注意*处如果包含家目录（home）不能写成~符号代替
#必须要写成"/home"的格式，否则会报错说找不到对应的目录
#读取的目录
read_path("*/grayvoc/trainval/VOCdevkit/VOC2007/JPEGImages")
#print(os.getcwd())
