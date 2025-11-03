import cv2
import numpy as np
import open3d as o3d
from utils.utils_metrics import compute_ADD, compute_ADD_S, compute_angel
import os
import json
from refiner_xh import refine
from data.lm.data import *
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
import sys

def compute_max_diameter(pcd):
    points = np.asarray(pcd.points)
    n = points.shape[0]
    max_distance = 0
    for i in range(n):
        for j in range(i + 1, n):
            distance = np.linalg.norm(points[i] - points[j])
            max_distance = max(max_distance, distance)
    return max_distance


def nocs_pnp(nocs: np.ndarray,magnitude, brightness_threshold=120,
             intr=np.array([[572.4114, 0.0, 325.2611], [0.0, 573.57043, 242.04899], [0.0, 0.0, 1.0]],
                           dtype=np.float32), index = None):

    minp = data_dict[f"{index:04d}"]["minp"]
    maxp = data_dict[f"{index:04d}"]["maxp"]

    points_2d = []
    points_3d = []

    img_show = np.zeros(nocs.shape)

    for x in range(nocs.shape[1]):
        for y in range(nocs.shape[0]):
            if sum(nocs[y, x]) >= brightness_threshold:
                if magnitude[y,x] < 40:
                    img_show[y, x] = 255
                    points_2d.append([x, y])
                    # print(nocs[y,x], nocs[y,x] / 255.0 * (maxp - minp) + minp)
                    points_3d.append(nocs[y,x] / 255.0 * (maxp - minp) + minp)


    points_2d = np.array(points_2d, dtype=np.float32)
    points_3d = np.array(points_3d, dtype=np.float32)

    # temp = points_3d
    # points_3d[0, :] = temp[2, :]
    # points_3d[1, :] = temp[1, :]
    # points_3d[2, :] = temp[0, :]

    # pcd = o3d.geometry.PointCloud()
    # pcd.points = o3d.utility.Vector3dVector(points_3d)
    # o3d.visualization.draw_geometries([pcd])

    # intr = np.array([[572.4114, 0.0, 325.2611], [0.0, 573.57043, 242.04899], [0.0, 0.0, 1.0]], dtype=np.float32)  # 设相机内参矩阵K
    distCoeffs = np.array([0, 0, 0, 0, 0], dtype=np.float32)
    retval, rvec, tvec, _ = cv2.solvePnPRansac(objectPoints=points_3d, imagePoints=points_2d, cameraMatrix=intr,
                                               distCoeffs=None, flags=cv2.SOLVEPNP_EPNP)
    # print(retval, rvec, tvec)

    R, _ = cv2.Rodrigues(rvec)

    T = np.eye(4)

    T[0:3, 0:3] = R

    T[0:3, 3] = tvec.flatten()
    return T

if __name__ == "__main__":
    parser = ArgumentParser(description="Training script parameters")
    parser.add_argument('--index', type=int, default=6)
    parser.add_argument('--debug', action='store_true', default=True)
    parser.add_argument('--gui', action='store_true', default=True)
    parser.add_argument('--add_s', action='store_true', default=False)
    args = parser.parse_args(sys.argv[1:])
    index = args.index
    debug = args.debug
    gui = args.gui
    adds_flag = args.add_s

    start_num = 0

    source_path = f"data/lm/test/{index:06d}/scene_gt.json"
    with open(source_path, 'r') as file:
        data = json.load(file)

    # data_item = data["3"]

    gt_Tw2c = []
    for key in data:
        cam_R_m2c = np.array(data[key][0]['cam_R_m2c']).reshape(3, 3)
        cam_t_m2c = np.array(data[key][0]['cam_t_m2c'])
        gt_T = np.eye(4)
        gt_T[:3, :3] = cam_R_m2c
        gt_T[:3, 3] = cam_t_m2c
        gt_Tw2c.append(gt_T)


    ply_path = f"data/lm/models/obj_{index:06d}.ply"
    point_cloud = o3d.io.read_point_cloud(ply_path)

    points = np.asarray(point_cloud.points)

    # max_diameter = compute_max_diameter(point_cloud)
    max_diameter = dia_dict[f"{index:04d}"]
    print(max_diameter)

    folder_path = f"result/{index:06d}_lm/"
    original_path = f"cut_result/{index:06d}_lm/"
    mask_path = f"data/lm/test/{index:06d}/mask_visib/"
    depth_path = f"data/lm/test/{index:06d}/depth/"
    ICP_ply_path = f"data/lm/test/{index:06d}/ply/"

    image_paths = sorted([os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.png')])
    original_paths = sorted([os.path.join(original_path, f) for f in os.listdir(original_path) if f.endswith('.png')])
    mask_paths = sorted([os.path.join(mask_path, f) for f in os.listdir(mask_path) if f.endswith('.png')])
    depth_paths = sorted([os.path.join(depth_path, f) for f in os.listdir(depth_path) if f.endswith('.png')])
    ICP_ply_paths= sorted([os.path.join(ICP_ply_path, f) for f in os.listdir(ICP_ply_path) if f.endswith('ply')])

    intr = np.array([[572.4114, 0.0, 325.2611], [0.0, 573.57043, 242.04899], [0.0, 0.0, 1.0]],
                    dtype=np.float32)
    pred_Tw2c = []
    count = 0

    angles = [3, 5, 8, 10, 13, 15, 18, 20, 22, 25, 27, 30, 33, 35, 40]
    corrrect_counts = {angle: 0 for angle in angles}
    wrong_counts = {angle: 0 for angle in angles}
    success_rates = {angle: -1 for angle in angles}

    for image_path in image_paths:
        # count_imgs += 1
        # if count_imgs <= start_num:
        #     continue


        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        grad_x = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)

        print(image_path)

        magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2)
        Tc2w = nocs_pnp(img,magnitude,index = index)

        R = Tc2w[0:3, 0:3]
        t = Tc2w[0:3, 3]+15
        Tw2c = np.zeros_like(Tc2w)
        Tw2c[0:3, 0:3] = R
        Tw2c[0:3, 3] = t
        Tw2c[3, 3] = 1
        pred_Tw2c.append(Tw2c)
        count += 1
        print(count)
        if debug == True and count == 30:
           break

    add_values = []
    add_s_values = []
    t_bias = []
    refine_5cm_num = 0
    refine_angle_num=0
    refine_juli_num= 0

    # 记录分类
    coarse_5 = 0
    coarse_10 = 0
    coarse_15 = 0
    coarse_20 = 0
    coarse_25 = 0
    coarse_30 = 0

    refine_5 = 0
    refine_10 = 0
    refine_15 = 0
    refine_20 = 0
    refine_25 = 0
    refine_30 = 0

    refine_1cm_num=0
    ADD_5 = 0
    ADD_2 = 0

    refine_1cm_5du = 0


    count_r = 0
    for pred_Tw, gt_Tw, original, mask, depth, ICP_ply_road, mask_center_road in zip(pred_Tw2c, gt_Tw2c, original_paths, mask_paths, depth_paths,ICP_ply_paths,mask_paths):
        count_r += 1
        if count_r < start_num:
            continue

        pred_t = pred_Tw[0:3, 3]
        pred_R = pred_Tw[0:3, 0:3]

        gt_t = gt_Tw[0:3, 3]
        #t_bias.append(pred_t - gt_t)

        coarse_TW = pred_Tw

        angel = compute_angel(pred_Tw, gt_Tw)
        pred_t = pred_Tw[0:3, 3]
        t_bias = pred_t - gt_t
        print(t_bias, angel)

        dist_coeffs = np.zeros(5)

        original_image = cv2.imread(original)
        original_image = cv2.cvtColor( original_image, cv2.COLOR_BGR2RGB)
        # print(gt_Tw)

        gt_t = gt_Tw[0:3, 3]
        gt_z = gt_t[2]
        print(gt_z)
        # target = o3d.io.read_point_cloud(ICP_ply_road)
        # gt_z = ICP_xh_1(target)

        pred_Tw = refine(image=original_image,new_matrix=pred_Tw, maskpath = mask, depthpath = depth, d_ply_path = ICP_ply_road, gt_depth = gt_z , gt_T = gt_Tw, index = index, gui = gui)
        # pred_Tw[0:3, 0:3] = pred_R

        pred_R1 = pred_Tw[:3, :3]
        pred_t1 = pred_Tw[:3, 3]
        # pred_t1[2] += 7
        print("refine result:", pred_Tw)

        transform_matrix = np.eye(4)
        transform_matrix[:3, :3] = pred_R1
        transform_matrix[:3, 3] = pred_t1

        pred_t = pred_Tw[0:3, 3]
        t_bias = pred_t - gt_t

        add = compute_ADD(pred_Tw, gt_Tw, points)
        add_values.append(add)
        print(add)

        if adds_flag == True:
            add_s = compute_ADD(pred_Tw, gt_Tw, points)
            add_s_values.append(add_s)
            print(add, add_s)

        coarse_angel = compute_angel(coarse_TW, gt_Tw)

        angel = compute_angel(pred_Tw, gt_Tw)



        if angel<=5:
            refine_angle_num = refine_angle_num+1
        if adds_flag == False and add / max_diameter <= 0.1:
            refine_juli_num = refine_juli_num+1
        elif adds_flag == True and add_s / max_diameter <= 0.1:
            refine_juli_num = refine_juli_num + 1
        if add < 50 :
            refine_5cm_num = refine_5cm_num+1
        if add < 10 :
            refine_1cm_num = refine_1cm_num+1
        if add / max_diameter <= 0.05:
            ADD_5+=1
        if add / max_diameter <= 0.02:
            ADD_2+=1

        if angel<=5 and add < 10:
            refine_1cm_5du+=1


        if angel<5:
            refine_5+=1
        if 5<angel<10:
            refine_10+=1
        if 10<angel<15:
            refine_15+=1
        if 15<angel<20:
            refine_20+=1
        if 20 < angel < 25:
            refine_25 += 1
        if 25 < angel < 30:
            refine_30 += 1

        if coarse_angel<5:
            coarse_5+=1
        if 5<coarse_angel<10:
            coarse_10+=1
        if 10<coarse_angel<15:
            coarse_15+=1
        if 15<coarse_angel<20:
            coarse_20+=1
        if 20 < coarse_angel < 25:
            coarse_25 += 1
        if 25 < coarse_angel < 30:
            coarse_30 += 1


        print("coarse est <5：", coarse_5, "5-10：", coarse_10, "10-15：", coarse_15, "15-20：", coarse_20,"20-25：", coarse_25, "25-30：", coarse_30)
        print("refine est <5：", refine_5, "5-10：", refine_10, "10-15：", refine_15, "15-20：", refine_20,
              "20-25：", refine_25, "25-30：", refine_30)
        print("X error：", t_bias[0], "Y error：",t_bias[1] , "Z error：",t_bias[2] ,"angle error：",angel)
        print(count_r,"ADD10:",refine_juli_num,"5：",refine_angle_num,"5cm:",refine_5cm_num, "5 1cm",refine_1cm_5du)
        print("ADD10_AR:", refine_juli_num/count_r, "5_AR:", refine_angle_num/count_r,"5cm_AR:", refine_5cm_num/count_r)
        print("ADD5_AR:",ADD_5/count_r, "ADD2_AR:",ADD_2/count_r,"1cm_AR", refine_1cm_num/count_r)


    mean_add = np.mean(add_values)
    mean_add_s = np.mean(add_s_values)


