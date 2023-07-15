from __future__ import absolute_import, division, print_function

import os
import sys
import glob
import argparse
import numpy as np
import PIL.Image as pil
import matplotlib as mpl
import matplotlib.cm as cm
import open3d as o3d

import torch
from torch import nn
from torchvision import transforms, datasets

import networks
from layers import disp_to_depth
import cv2
import heapq
from PIL import ImageFile
import time

ImageFile.LOAD_TRUNCATED_IMAGES = True


def parse_args():
    parser = argparse.ArgumentParser(
        description='Simple testing function for Lite-Mono models.')

    parser.add_argument('--image_path', type=str,
                        default="D:/dataset/wass2k/planefit/input1",
                        # default="D:/dataset/wass2k/undist_resize/input1",
                        # default="D:/dataset/wass2k/special/input1",
                        # default="D:/dataset/wass2k/test/input1",
                        help='path to a test image or folder of images',
                        )

    parser.add_argument('--load_weights_folder', type=str,
                        default="tmp/lite-mono/models/weights_199",
                        help='path of a pretrained model to use',
                        )

    parser.add_argument('--test',
                        action='store_true',
                        help='if set, read images from a .txt file',
                        )

    parser.add_argument('--model', type=str,
                        help='name of a pretrained model to use',
                        default="lite-mono",
                        choices=[
                            "lite-mono",
                            "lite-mono-small",
                            "lite-mono-tiny"])

    parser.add_argument('--ext', type=str,
                        help='image extension to search for in folder', default="jpg")
    parser.add_argument("--no_cuda",
                        help='if set, disables CUDA',
                        action='store_true')

    return parser.parse_args()


def test_simple(args):
    """Function to predict for a single image or folder of images
    """
    assert args.load_weights_folder is not None, \
        "You must specify the --load_weights_folder parameter"

    if torch.cuda.is_available() and not args.no_cuda:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print("-> Loading model from ", args.load_weights_folder)
    encoder_path = os.path.join(args.load_weights_folder, "encoder.pth")
    decoder_path = os.path.join(args.load_weights_folder, "depth.pth")
    predictive_path = os.path.join(args.load_weights_folder, "predictive_mask.pth")

    encoder_dict = torch.load(encoder_path)
    decoder_dict = torch.load(decoder_path)
    predictive_dict = torch.load(predictive_path)
    # extract the height and width of image that this model was trained with
    feed_height = encoder_dict['height']
    feed_width = encoder_dict['width']

    # LOADING PRETRAINED MODEL
    print("   Loading pretrained encoder")
    encoder = networks.LiteMono(model=args.model,
                                height=feed_height,
                                width=feed_width)

    model_dict = encoder.state_dict()
    encoder.load_state_dict({k: v for k, v in encoder_dict.items() if k in model_dict})

    encoder.to(device)
    encoder.eval()

    print("   Loading pretrained decoder")
    depth_decoder = networks.DepthDecoder(encoder.num_ch_enc, scales=range(3))
    depth_model_dict = depth_decoder.state_dict()
    depth_decoder.load_state_dict({k: v for k, v in decoder_dict.items() if k in depth_model_dict})

    depth_decoder.to(device)
    depth_decoder.eval()

    print("   Loading predictive decoder")
    predictive_decoder = networks.DepthDecoder(encoder.num_ch_enc, scales=range(3))
    predictive_model_dict = predictive_decoder.state_dict()
    predictive_decoder.load_state_dict({k: v for k, v in predictive_dict.items() if k in predictive_model_dict})

    predictive_decoder.to(device)
    predictive_decoder.eval()

    # FINDING INPUT IMAGES
    # if os.path.isfile(args.image_path) and not args.test:
    #     # Only testing on a single image
    #     paths = [args.image_path]
    #     output_directory = os.path.dirname(args.image_path)
    # elif os.path.isfile(args.image_path) and args.test:
    #     gt_path = os.path.join('splits', 'eigen', "gt_depths.npz")
    #     gt_depths = np.load(gt_path, fix_imports=True, encoding='latin1', allow_pickle=True)["data"]
    #
    #     side_map = {"2": 2, "3": 3, "l": 2, "r": 3}
    #     # reading images from .txt file
    #     paths = []
    #     with open(args.image_path) as f:
    #         filenames = f.readlines()
    #         for i in range(len(filenames)):
    #             filename = filenames[i]
    #             line = filename.split()
    #             folder = line[0]
    #             if len(line) == 3:
    #                 frame_index = int(line[1])
    #                 side = line[2]
    #
    #             f_str = "{:010d}{}".format(frame_index, '.jpg')
    #             image_path = os.path.join(
    #                 'kitti_data',
    #                 folder,
    #                 "image_0{}/data".format(side_map[side]),
    #                 f_str)
    #             paths.append(image_path)
    #
    # elif os.path.isdir(args.image_path):
    #     # Searching folder for images
    #     paths = glob.glob(os.path.join(args.image_path, '*.{}'.format(args.ext)))
    #     output_directory = args.image_path
    # else:
    #     raise Exception("Can not find args.image_path: {}".format(args.image_path))
    output_directory = args.image_path
    fpath = os.path.join(args.image_path)
    paths = os.listdir(fpath)
    paths.sort(key=lambda x: int(x.split('.')[0]))
    print("-> Predicting on {:d} test images".format(len(paths)))
    ones = torch.ones([1, 1, feed_height, feed_width]).cuda()
    zeros = torch.zeros([1, 1, feed_height, feed_width]).cuda()
    max_pool = nn.MaxPool2d(kernel_size=7, stride=1, padding=3)  # 可调整 kernel_size 和 padding

    u_coord = torch.linspace(0, feed_width - 1, steps=feed_width)
    u_coords = u_coord.repeat(feed_height, 1).unsqueeze(0).unsqueeze(1).cuda()
    u_coords_1 = u_coords.reshape((1, feed_height * feed_width))
    u_coord_s = (u_coords + 1 - 222.7768) / 365.1978

    v_coord = torch.linspace(0, feed_height - 1, steps=feed_height).reshape((feed_height, 1))
    v_coords = v_coord.repeat(1, feed_width).unsqueeze(0).unsqueeze(1).cuda()
    v_coords_1 = v_coords.reshape((1, feed_height * feed_width))

    v_coord_s = (v_coords + 1 - 273.7560) / 364.3931

    # Q = torch.Tensor([[1., 0., 0., -271.88163376],
    #                   [0., 1., 0., -222.17257118],
    #                   [0., 0., 0., 364.7891992],
    #                   [0., 0., 1., -0.]]).cuda()
    Q = torch.Tensor([[1., 0., 0., -271.88163376],
                      [0., 1., 0., -222.17257118],
                      [0., 0., 0., 364.7891992],
                      [0., 0., 0.39292731, -0.]]).cuda()

    ones_1 = torch.ones([1, feed_height * feed_width]).cuda()
    # PREDICTING ON EACH IMAGE IN TURN
    plane_list = []

    z_t = torch.zeros((2500, 20)).cuda()
    time_start = time.time()
    with torch.no_grad():
        for idx, image_path in enumerate(paths):

            if image_path.endswith("_disp.jpg"):
                # don't try to predict disparity for a disparity image!
                continue

            # if idx > 10:
            #     break
            img_path = os.path.join(args.image_path, image_path)
            # Load image and preprocess
            input_image = pil.open(img_path).convert('L')
            original_width, original_height = input_image.size
            input_image = input_image.resize((feed_width, feed_height), pil.LANCZOS)
            input_image = transforms.ToTensor()(input_image).unsqueeze(0)

            # PREDICTION
            input_image = input_image.to(device)

            # torch.cuda.synchronize()
            # time_start = time.time()

            features = encoder(input_image)
            outputs = depth_decoder(features)

            mask_outputs = predictive_decoder(features)
            disp_l = outputs[("disp", 0, 0)]
            # disp_r = outputs[("disp", 's', 0)]
            mask_l = mask_outputs[("disp", 0, 0)]
            # mask_r = mask_outputs[("disp", 's', 0)]

            # 掩膜处理
            mask_l = torch.where(mask_l <= 0.27, zeros, ones)
            # mask_r = torch.where(mask_r <= 0.27, zeros, ones)
            # 假设要处理的张量为 tensor
            # ===== 膨胀 =====
            tensor_dilate = max_pool(mask_l)
            # ===== 腐蚀 =====
            tensor_erode1 = -max_pool(-tensor_dilate)
            tensor_erode2 = -max_pool(-tensor_erode1)
            # # # #
            # # # # mask_save = tensor_erode2.detach().cpu().squeeze()
            # # # # np.savetxt("mask{:d}_save.txt".format(idx), mask_save, fmt='%f')

            # # 掩膜+输入图
            # input_image_r = (255 * input_image)
            # mask_img = input_image_r * tensor_erode2
            # mask_cpu = mask_img.detach().cpu().squeeze().numpy()
            # mask_save = mask_cpu.astype(np.uint8)
            #
            # cv2.imwrite("mask_save.png", mask_save)

            # # roi视差图
            # # disp_l_mask = disp_l * tensor_erode2
            # disp_l_mask = disp_l
            # disp_roi = disp_l_mask.detach().cpu().squeeze()
            # np.savetxt("disp_mono{:d}.txt".format(idx), disp_roi, fmt='%f')
            # continue
            # cv2.imshow("disp_roi", disp_roi.numpy())
            # cv2.waitKey(0)

            # 通过Q矩阵将视差图转化为点云
            displ_width = feed_width * (disp_l.reshape((1, feed_height * feed_width))) - 1
            img_coords = torch.cat((u_coords_1, v_coords_1, displ_width, ones_1), dim=0)
            cam_coords = Q @ img_coords
            tmp = cam_coords.clone()
            cam_coords[:, :] /= tmp[3, :]
            # Xw = cam_coords[0, :].reshape((1, 1, feed_height, feed_width))
            # Yw = cam_coords[1, :].reshape((1, 1, feed_height, feed_width))
            # Zw = cam_coords[2, :].reshape((1, 1, feed_height, feed_width))

            # # 原始海浪点云,rgb映射
            # # Zw_mask = 2.545 * Zw * tensor_erode2
            # # Xw_mask = 2.545 * Xw * tensor_erode2
            # # Yw_mask = 2.545 * Yw * tensor_erode2
            # input_image_r = (255 * input_image).reshape([feed_width * feed_height, 1])
            #
            # point_mesh = torch.cat(
            #     (Xw.reshape([feed_width * feed_height, 1]),
            #      Yw.reshape([feed_width * feed_height, 1]),
            #      Zw.reshape([feed_width * feed_height, 1]),
            #      input_image_r,
            #      input_image_r,
            #      input_image_r), axis=1)
            # point_cloud = point_mesh.cpu().numpy()
            # np.savetxt('point_cloud1070.txt', point_cloud, fmt='%f')
            # 视差图直接转化
            # # Zw = 364.3931 / (disp_l * feed_width)
            # Zw = 0.66984025 / (disp_l - 0.00183823)
            # Xw = v_coord_s * Zw
            # Yw = u_coord_s * Zw

            # # # 平面拟合获取每幅图的平面参数
            # Zw_mask = 2.545 * Zw * tensor_erode2
            # Xw_mask = 2.545 * Xw * tensor_erode2
            # Yw_mask = 2.545 * Yw * tensor_erode2
            #
            # # # point_cloud_name = os.path.join(save_path, 'point_cloud{:n}.pcd'.format(i))
            # #
            # point_mesh = torch.cat((Xw_mask.reshape([feed_width * feed_height, 1]),
            #                         Yw_mask.reshape([feed_width * feed_height, 1]),
            #                         Zw_mask.reshape([feed_width * feed_height, 1])), axis=1)
            # pcd = o3d.geometry.PointCloud()
            #
            # pcd.points = o3d.utility.Vector3dVector(point_mesh.cpu().numpy())
            #
            # # o3d.io.write_point_cloud("point_cloud_q{:n}.pcd".format(idx), pcd, write_ascii=True)
            #
            # plane_model, inliers = pcd.segment_plane(distance_threshold=0.25,
            #                                          ransac_n=3,
            #                                          num_iterations=1000)
            # # [a, b, c, d] = plane_model
            # plane_list.append(plane_model)
            # continue

            #
            # # pcd.colors = o3d.utility.Vector3dVector(point_mesh[:, 3:])
            #
            # o3d.io.write_point_cloud("point_cloud{:n}.pcd".format(idx), pcd, write_ascii=True)

            # X_s = 0.70920548 * Xw - 0.00112136 * Yw + -0.70500094 * Zw
            # Y_s = 0.00112136 * Xw - 0.99999568 * Yw + 0.00271863 * Zw
            # Z_s = 0.70148106 * Xw - 0.0197351 * Yw + 0.7124148 * Zw - 4.470256483999997

            # plane_new with disp-1 (05/16)
            # X_s = 0.7070764 * Xw - 0.00825033 * Yw - 0.70708903 * Zw
            # Y_s = -0.00825033 * Xw + 0.99976763 * Yw - 0.0199155 * Zw
            # Z_s = 0.70708903 * Xw + 0.0199155 * Yw + 0.70684403 * Zw - 4.512754291999997

            # plane_new with Q (05/17) 155组
            # X_s = 0.99966726 * Xw - 0.01060162 * Yw - 0.02351562 * Zw
            # Y_s = -0.01060162 * Xw + 0.66222016 * Yw - 0.74923432 * Zw
            # Z_s = 0.02351562 * Xw + 0.74923432 * Yw + 0.66188742 * Zw - 4.809079129032257

            # 250组
            # Z_s = 0.02100126 * Xw + 0.75008342 * Yw + 0.66100968 * Zw - 4.786501683999999

            # 250组 d-0.5
            # Z_s = 0.02100126 * Xw + 0.75008342 * Yw + 0.66100968 * Zw - 4.786501683999999

            Pcam2sea = torch.Tensor([[0.99972316, - 0.00978791, - 0.02139612, 0],
                                     [-0.00978791, 0.65393668, - 0.75648597, 0],
                                     [0.02139612, 0.75648597, 0.65365984, -12.285575211999998],
                                     [0., 0., 0., 1]]).cuda()
            # seapoints = (Pcam2sea @ cam_coords).T
            seapoints = cam_coords.T
            input_image_r = (255 * input_image).reshape([feed_width * feed_height, 1])
            mask_map = tensor_erode2.reshape([feed_width * feed_height, 1])

            Xw_mask = torch.mul(seapoints[:, 0].reshape([feed_width * feed_height, 1]) , mask_map)
            Yw_mask = seapoints[:, 1].reshape([feed_width * feed_height, 1]) * mask_map
            Zw_mask = seapoints[:, 2].reshape([feed_width * feed_height, 1]) * mask_map

            point_mesh = torch.cat(
                (Xw_mask,
                 Yw_mask,
                 Zw_mask,
                 input_image_r,
                 input_image_r,
                 input_image_r), axis=1)

            point_cloud = point_mesh.cpu().numpy()
            np.save('715cam_point_cloud.npy', point_cloud)
            np.savetxt('715cam_point_cloud.txt', point_cloud, fmt='%f')
            continue
            # 250组 d-1
            X_s = 0.99972316 * Xw - 0.00978791 * Yw - 0.02139612 * Zw
            Y_s = -0.00978791 * Xw + 0.65393668 * Yw - 0.75648597 * Zw
            Z_s = 0.02139612 * Xw + 0.75648597 * Yw + 0.65365984 * Zw - 4.828070316000002

            # Z_s = 0.02351562 * Xw + 0.74923432 * Yw + 0.66188742 * Zw -4.809079129032257
            # # 确定点云最大z高度和位置
            # Zs_mask_tmp = (Z_s * tensor_erode2).squeeze()
            #
            # Zs_mask = 2.5 * Zs_mask_tmp
            # Z_max = -torch.min(Zs_mask)
            #
            # a_col = torch.min(Zs_mask, dim=0)
            # a_row = torch.min(Zs_mask, dim=1)
            #
            # b_col = torch.argmin(a_col[0], dim=0)
            # b_row = torch.argmin(a_row[0], dim=0)
            #
            # print("z_max:", Z_max.cpu(), "pix_cood:", b_row.cpu(), ",", b_col.cpu())
            # continue

            # # 探针设置
            # Z_s_1 = -2.545 * (Z_s.squeeze())
            # if idx < 2500:
            #     z_t_tmp = torch.Tensor(
            #         [Z_s_1[70, 328], Z_s_1[142, 132], Z_s_1[176, 335], Z_s_1[116, 320], Z_s_1[58, 384],
            #          Z_s_1[185, 134], Z_s_1[160, 194], Z_s_1[208, 180], Z_s_1[120, 386], Z_s_1[71, 135],
            #          Z_s_1[154, 357], Z_s_1[147, 475], Z_s_1[112, 154], Z_s_1[92, 193], Z_s_1[348, 382],
            #          Z_s_1[142, 365], Z_s_1[158, 383], Z_s_1[58, 268], Z_s_1[70, 453], Z_s_1[186, 335],
            #          ])
            #     z_t[idx, :] = z_t_tmp
            # else:
            #     break
            # continue

            Zw_mask = -2.545 * Z_s * tensor_erode2
            Zw_mask_cpu = Zw_mask.squeeze().cpu().numpy()
            Zw_mask_cpu[Zw_mask_cpu == 0] = np.nan
            np.savetxt('sea526.txt', Zw_mask_cpu, fmt='%f')

            Xw_mask = 2.545 * X_s * tensor_erode2
            Yw_mask = 2.545 * Y_s * tensor_erode2
            input_image_r = (255 * input_image).reshape([feed_width * feed_height, 1])

            point_mesh = torch.cat(
                (Xw_mask.reshape([feed_width * feed_height, 1]),
                 Yw_mask.reshape([feed_width * feed_height, 1]),
                 Zw_mask.reshape([feed_width * feed_height, 1]),
                 input_image_r,
                 input_image_r,
                 input_image_r), axis=1)

            point_cloud = point_mesh.cpu().numpy()
            np.save('713sea_point_cloud.npy', point_cloud)
            np.savetxt('713sea_point_cloud.txt', point_cloud, fmt='%f')

            # pcd = o3d.geometry.PointCloud()
            #
            # pcd.points = o3d.utility.Vector3dVector(point_mesh.cpu().numpy())
            #
            # # pcd.colors = o3d.utility.Vector3dVector(point_mesh[:, 3:])
            #
            # o3d.io.write_point_cloud("point_cloud{:n}.pcd".format(idx), pcd, write_ascii=True)
            continue

            z_map = Zw_mask.detach().cpu().squeeze()
            # max_z = z_map.max()
            # min_z = z_map.min()
            # print(max_z)
            # print(min_z)
            # z_range = max_z - min_z
            z_map_img = 255 * (z_map - 0.6) / 1.2
            # print(z_range)
            im_color = cv2.applyColorMap(cv2.convertScaleAbs(z_map_img.numpy().astype(np.uint8), alpha=1),
                                         cv2.COLORMAP_JET)
            # cv2.imshow("im_color", im_color)
            cv2.imwrite("D:/dataset/wass2k/val/depth_img/depth_color{:d}.png".format(idx), im_color)
            # cv2.waitKey(0)

            continue
            # torch.cuda.synchronize()
            # time_end = time.time()
            # time_sum = time_end - time_start
            # print(time_sum)
            # continue
            # cv2.namedWindow("disp_roi", cv2.WINDOW_FREERATIO)
            # cv2.imshow("disp_roi", disp_roi.numpy())
            # cv2.waitKey(0)

            displ_resized = torch.nn.functional.interpolate(
                disp_l, (original_height, original_width), mode="bilinear", align_corners=False)
            dispr_resized = torch.nn.functional.interpolate(
                disp_r, (original_height, original_width), mode="bilinear", align_corners=False)
            maskl_resized = torch.nn.functional.interpolate(
                mask_l, (original_height, original_width), mode="bilinear", align_corners=False)
            mask_resized = torch.nn.functional.interpolate(
                mask_r, (original_height, original_width), mode="bilinear", align_corners=False)

            # Saving numpy file
            # output_name = os.path.splitext(os.path.basename(image_path))[0]
            # output_name = os.path.splitext(image_path)[0].split('/')[-1]
            # scaled_disp, depth = disp_to_depth(disp, 0.1, 100)

            name_dest_npy1 = os.path.join(output_directory, "displ_resized{}.npy".format(idx))
            np.save(name_dest_npy1, displ_resized.cpu().numpy())
            name_dest_npy2 = os.path.join(output_directory, "dispr_resized{}.npy".format(idx))
            np.save(name_dest_npy2, dispr_resized.cpu().numpy())
            name_dest_npy3 = os.path.join(output_directory, "maskl_resized{}.npy".format(idx))
            np.save(name_dest_npy3, maskl_resized.cpu().numpy())
            name_dest_npy4 = os.path.join(output_directory, "maskr_resized{}.npy".format(idx))
            np.save(name_dest_npy4, mask_resized.cpu().numpy())

            # Saving colormapped depth image
            # disp_resized_np = disp_resized.squeeze().cpu().numpy()
            # vmax = np.percentile(disp_resized_np, 95)
            # normalizer = mpl.colors.Normalize(vmin=disp_resized_np.min(), vmax=vmax)
            # mapper = cm.ScalarMappable(norm=normalizer, cmap='magma')
            # colormapped_im = (mapper.to_rgba(disp_resized_np)[:, :, :3] * 255).astype(np.uint8)
            # im = pil.fromarray(colormapped_im)
            #
            # name_dest_im = os.path.join(output_directory, "{}_disp.jpeg".format(output_name))
            # im.save(name_dest_im)
            #
            # print("   Processed {:d} of {:d} images - saved predictions to:".format(
            #     idx + 1, len(paths)))
            # print("   - {}".format(name_dest_im))
            # print("   - {}".format(name_dest_npy))
    time_end = time.time()
    sum_t = (time_end - time_start)  # 运行所花时间
    print('time cost', sum_t, 's')
    print('-> Done!')
    # result1 = np.array(z_t.cpu())
    # np.savetxt('714plane_fit.txt', result1, fmt='%f')

    result1 = np.array(plane_list)
    np.savetxt('714plane_fit.txt', result1, fmt='%f')


if __name__ == '__main__':
    args = parse_args()
    test_simple(args)
