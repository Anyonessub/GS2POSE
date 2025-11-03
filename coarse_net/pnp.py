import cv2
import numpy as np
import open3d as o3d
from itertools import combinations
from scene.dataset_readers import sceneLoadTypeCallbacks
from utils.utils_metrics import compute_ADD, compute_ADD_S, compute_angel
import os
import json

def nocs_pnp(nocs_path, intr):
    # 假设rgb_image是RGB彩色图像，points_2d是图像上的点坐标，points_3d是对应的物体坐标系下的点坐标
    # 假设K是相机内参矩阵
    rgb_image = cv2.imread(nocs_path,cv2.COLOR_BGR2RGB)
    # rgb_image = color_filter(rgb_image)
    height, width, _ = rgb_image.shape
    center = (width // 2, height // 2)
    radius = min(width, height) // 3  # 选择中心到边缘的1/3作为半径范围
    points_2d = []
    points_3d = []
    brightness_threshold = 500
    for y in range(rgb_image.shape[0]):
        for x in range(rgb_image.shape[1]):
            pixel = rgb_image[y, x]
            brightness = sum(pixel)  # 计算像素的亮度
            if brightness > brightness_threshold:

                points_2d.append((x, y))
                # print(pixel)
                points_3d.append((pixel / 255 - 0.5) * 94.4092)


    points_2d = np.array(points_2d, dtype=np.float32)
    points_3d = np.array(points_3d, dtype=np.float32)

    # intr = np.array([[572.4114, 0.0, 325.2611], [0.0, 573.57043, 242.04899], [0.0, 0.0, 1.0]], dtype=np.float32)  # 设相机内参矩阵K
    distCoeffs = np.array([0, 0, 0, 0, 0], dtype=np.float32)
    # 使用solvePnP算法估计位姿变换矩阵T
    retval, rvec, tvec = cv2.solvePnP(objectPoints=points_3d, imagePoints=points_2d, cameraMatrix=intr, distCoeffs=distCoeffs)

    # 将旋转向量转换为旋转矩阵
    R, _ = cv2.Rodrigues(rvec)

    # 构造位姿变换矩阵T
    # 创建一个4x4的单位矩阵
    T = np.eye(4)

    # 将旋转矩阵 R 放入变换矩阵 T 的左上角
    T[0:3, 0:3] = R

    # 将平移向量 tvec 放入变换矩阵 T 的最后一列
    T[0:3, 3] = tvec.flatten()
    return T

if __name__ == "__main__":

    source_path = "output/94f7a89c-1/cameras.json"
    with open(source_path, 'r') as file:
        data = json.load(file)

    gt_Tw2c = []
    # 依次读取每个旋转矩阵和平移向量
    for i, camera in enumerate(data):
        cam_R_m2c = np.array(camera["rotation"])
        cam_t_m2c = np.array(camera["position"])
        gt_T = np.eye(4)
        gt_T[:3, :3] = cam_R_m2c
        gt_T[:3, 3] = cam_t_m2c
        gt_Tw2c.append(gt_T)

    # 读取PLY文件
    ply_path = "data/lm/models/obj_000001.ply"
    point_cloud = o3d.io.read_point_cloud(ply_path)

    # 从点云中提取3D点坐标
    points = np.asarray(point_cloud.points)
    #distances = [np.linalg.norm(p1 - p2) for p1, p2 in combinations(points, 2)]

    # 指定文件夹路径
    folder_path = "output/94f7a89c-1/train/ours_5000/renders"

    # 获取文件夹中所有PNG图片的路径，并按照文件顺序排序
    image_paths = sorted([os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.png')])
    intr = np.array([[572.4114, 0.0, 325.2611], [0.0, 573.57043, 242.04899], [0.0, 0.0, 1.0]],
                    dtype=np.float32)  # 设相机内参矩阵K
    pred_Tw2c = []
    count = 0
    addcount = 0
    addscount = 0
    addcount_10 = 0
    addscount_10 = 0
    # 依次处理每张图片
    for image_path in image_paths:
        # 调用nocs_pnp函数，传入图片路径
        Tc2w = nocs_pnp(image_path, intr)

        # 将Tc2w转换为Tw2c
        # 将Tw2c转换为Tc2w的方法
        R = Tc2w[0:3, 0:3]  # 提取旋转矩阵部分
        t = Tc2w[0:3, 3]  # 提取平移向量部分
        Tw2c = np.zeros_like(Tc2w)  # 初始化Tc2w矩阵
        Tw2c[0:3, 0:3] = -np.fliplr(R)  # 计算旋转矩阵的转置
        Tw2c[0:3, 3] = -t  # 计算平移向量
        Tw2c[3, 3] = 1  # 设置最后一个元素为1
        pred_Tw2c.append(Tw2c)
        count += 1
        print(count)
        if count == 20:
            break

    # 计算ADD与ADD-S值并求平均
    add_values = []
    add_s_values = []
    count = 0
    for pred_Tw, gt_Tw in zip(pred_Tw2c, gt_Tw2c):
        print(pred_Tw)
        print(gt_Tw)
        add = compute_ADD(pred_Tw, gt_Tw, points)
        add_s = compute_ADD_S(pred_Tw, gt_Tw, points)
        angel = compute_angel(pred_Tw, gt_Tw)
        add_values.append(add)
        add_s_values.append(add_s)
        if add<50 and angel<5:
            addcount+=1
        if add_s<50 and angel<5:
            addscount+=1
        print(add,add_s,angel)
        count += 1
        print(count)


    mean_add = np.mean(add_values)
    mean_add_s = np.mean(add_s_values)

    print("平均ADD值：", mean_add)
    print("平均ADD-S值：", mean_add_s)
    print("ADD AR:", addcount)
    print("ADD-S AR:", addscount)
    print("ADD-10 AR:", addcount_10)
    print("ADD-S-10 AR:", addscount_10)
