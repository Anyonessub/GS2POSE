import cv2
import numpy as np
import open3d as o3d
import OpenEXR
from torch.utils.tensorboard.summary import image

from utils.utils_metrics import compute_ADD, compute_ADD_S, compute_angel
import os
import json
from refiner_xh import refine
from ICP import ICP_xh

def compute_max_diameter(pcd):
    # 获取点云中的所有点
    points = np.asarray(pcd.points)
    n = points.shape[0]
    max_distance = 0
    for i in range(n):
        for j in range(i + 1, n):
            distance = np.linalg.norm(points[i] - points[j])
            max_distance = max(max_distance, distance)
    return max_distance

def read_exr_image(file_path):
    """Read an EXR image and return it as a numpy array in RGB format."""

    # 打开 EXR 文件
    exr_file = OpenEXR.InputFile(file_path)

    # 获取图像的大小
    dw = exr_file.header()['dataWindow']
    width = dw.max.x - dw.min.x + 1
    height = dw.max.y - dw.min.y + 1

    # 读取 RGB 通道
    channels = ['R', 'G', 'B']
    rgb = [np.frombuffer(exr_file.channel(c), dtype=np.float32) for c in channels]

    # 将通道转换为 (height, width, channels) 形状
    img = np.stack(rgb, axis=-1).reshape(height, width, 3)

    # 关闭 EXR 文件
    exr_file.close()

    return img

def nocs_pnp(nocs: np.ndarray,magnitude, brightness_threshold=120,
             intr=np.array([[572.4114, 0.0, 325.2611], [0.0, 573.57043, 242.04899], [0.0, 0.0, 1.0]],
                           dtype=np.float32)):
    #0001
    # minp = np.asarray([-38.7771, -39.6638, -47.6449])
    # maxp = np.asarray([38.7483, 39.5534, 46.5404])
    #0002
    minp = np.asarray([-41.2807, -23.5249, -41.7435])
    maxp = np.asarray([41.4265, 23.5842, 41.6361])
    #0003
    # minp = np.asarray([-31.9354, -31.7139, -14.5107])
    # maxp = np.asarray([31.9436, 31.6627, 14.3547])
    #0004
    # minp = np.asarray([-26.2880, -27.5745, -19.4907])
    # maxp = np.asarray([26.5324, 27.6799, 19.2426])
    #0005
    # minp = np.asarray([-19.5433, -34.8125, -39.8893])
    # maxp = np.asarray([19.5443, 34.9029, 36.8508])
    #0006
    # minp = np.asarray([-13.0093, -24.7142, -22.4430])
    # maxp = np.asarray([13.0736, 24.6765, 22.3916])

    """
    intr相机内参,minp，maxp模型尺度信息
    利用NOCS图像, 对图像中的模型进行位姿估计

    nocs: 以RGB图像保存的三通道ndarray
    """
    points_2d = []
    points_3d = []

    img_show = np.zeros(nocs.shape)

    for x in range(nocs.shape[1]):
        for y in range(nocs.shape[0]):
            if sum(nocs[y, x]) >= brightness_threshold:
                if magnitude[y,x] < 20:
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
    # 使用solvePnP算法估计位姿变换矩阵T
    retval, rvec, tvec, _ = cv2.solvePnPRansac(objectPoints=points_3d, imagePoints=points_2d, cameraMatrix=intr,
                                               distCoeffs=None, flags=cv2.SOLVEPNP_EPNP)
    # print(retval, rvec, tvec)

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
    start_num = 230
    num = 10

    source_path = "data/lm/test/000002/scene_gt.json"
    with open(source_path, 'r') as file:
        data = json.load(file)

    gt_Tw2c = []
    # 依次读取每个旋转矩阵和平移向量
    for key in data:
        cam_R_m2c = np.array(data[key][0]['cam_R_m2c']).reshape(3, 3)
        cam_t_m2c = np.array(data[key][0]['cam_t_m2c'])
        gt_T = np.eye(4)
        gt_T[:3, :3] = cam_R_m2c
        gt_T[:3, 3] = cam_t_m2c
        gt_Tw2c.append(gt_T)

    # 读取PLY文件
    ply_path = "data/lm/models/obj_000002.ply"
    point_cloud = o3d.io.read_point_cloud(ply_path)

    # 从点云中提取3D点坐标
    points = np.asarray(point_cloud.points)

    # max_diameter = compute_max_diameter(point_cloud)
    max_diameter = 247.506
    print(max_diameter)

    # 指定文件夹路径
    folder_path = "result/000002/"
    original_path = "cut_result/000002/"
    mask_path = "data/lm/test/000002/mask_visib/"
    depth_path = "data/lm/test/000002/depth_mask/"
    ICP_ply_path = "data/lm/test/000002/ply/"

    # 获取文件夹中所有PNG图片的路径，并按照文件顺序排序
    image_paths = sorted([os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.png')])
    original_paths = sorted([os.path.join(original_path, f) for f in os.listdir(original_path) if f.endswith('.png')])
    mask_paths = sorted([os.path.join(mask_path, f) for f in os.listdir(mask_path) if f.endswith('.png')])
    depth_paths = sorted([os.path.join(depth_path, f) for f in os.listdir(depth_path) if f.endswith('.png')])
    ICP_ply_paths= sorted([os.path.join(ICP_ply_path, f) for f in os.listdir(ICP_ply_path) if f.endswith('ply')])

    intr = np.array([[572.4114, 0.0, 325.2611], [0.0, 573.57043, 242.04899], [0.0, 0.0, 1.0]],
                    dtype=np.float32)  # 设相机内参矩阵K
    pred_Tw2c = []
    count = 0
    addcount = 0
    addscount = 0
    addcount_10 = 0
    addscount_10 = 0
    refine_5cm_num = 0
    # 依次处理每张图片
    count_imgs = 0
    # for ICP_ply_road in ICP_ply_paths:
    #     count_imgs += 1
    #
    #     target = o3d.io.read_point_cloud(ICP_ply_road)
    #     R,t = ICP_xh(target)
    #     Tw2c = np.empty((4, 4)) # 初始化Tc2w矩阵
    #     Tw2c[0:3, 0:3] = R  # 计算旋转矩阵的转置
    #     Tw2c[0:3, 3] = t  # 计算平移向量
    #     Tw2c[3, 3] = 1  # 设置最后一个元素为1
    #     pred_Tw2c.append(Tw2c)
    #     count += 1
    #     print(count)
    #     if count == 20:
    #        break

    #采用NOCS图像估计
    for image_path in image_paths:
        count_imgs += 1
        # if count_imgs <= start_num:
        #     continue
        # 调用nocs_pnp函数，传入图片路径
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        #计算梯度数值
        gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 使用Sobel算子计算x方向和y方向上的梯度
        grad_x = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)

        # 计算梯度幅值
        magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2)
        Tc2w = nocs_pnp(img,magnitude)

        # 将Tc2w转换为Tw2c
        # 将Tw2c转换为Tc2w的方法
        R = Tc2w[0:3, 0:3]  # 提取旋转矩阵部分
        t = Tc2w[0:3, 3]  # 提取平移向量部分
        Tw2c = np.zeros_like(Tc2w)  # 初始化Tc2w矩阵
        Tw2c[0:3, 0:3] = R  # 计算旋转矩阵的转置
        Tw2c[0:3, 3] = t  # 计算平移向量
        Tw2c[3, 3] = 1  # 设置最后一个元素为1
        pred_Tw2c.append(Tw2c)
        count += 1
        print(count)
        # if count == 20:
        #    break

    # 计算ADD与ADD-S值并求平均
    add_values = []
    add_s_values = []
    t_bias = []
    count = 0
    angle_num=0
    refine_angle_num=0
    juli_num=0
    refine_juli_num=0
    count_over = 0
    wrong_folder= "./wrong/"

    #精估网络部分代码
    for pred_Tw, gt_Tw, original, mask, depth, ICP_ply_road in zip(pred_Tw2c, gt_Tw2c, original_paths, mask_paths, depth_paths,ICP_ply_paths):

        #粗估网络部分
        # print(gt_Tw)
        pred_t = pred_Tw[0:3, 3]
        gt_t = gt_Tw[0:3, 3]
        #t_bias.append(pred_t - gt_t)
        add = compute_ADD(pred_Tw, gt_Tw, points)
        add_s = compute_ADD_S(pred_Tw, gt_Tw, points)
        angel = compute_angel(pred_Tw, gt_Tw)
        add_values.append(add)
        add_s_values.append(add_s)
        if add < 10:
            addcount += 1
        if add_s < 10:
            addscount += 1
        if add / max_diameter < 0.1:
            addcount_10 += 1
        if add_s / max_diameter < 0.1:
            addscount_10 += 1
        # if angel >= 35 or add >= 2 * max_diameter :
        #     continue
        count += 1
        # if count <=start_num:
        #     continue
            # count_over += 1
        print("粗估结果:", pred_Tw)  # 输入精估网络
        print("真值:", gt_Tw)
        print(add, add_s, angel)
        if angel < 10:
            angle_num = angle_num + 1
        if add < 10 or add_s < 10:
            juli_num = juli_num + 1

        # print(count, juli_num, angle_num)

        #图像显示部分
        dist_coeffs = np.zeros(5)

        #精估网络部分
        original_image = cv2.imread(original)
        original_image = cv2.cvtColor( original_image, cv2.COLOR_BGR2RGB)
        # print(gt_Tw)

        gt_t = gt_Tw[0:3, 3]
        gt_z = gt_t[2]
        target = o3d.io.read_point_cloud(ICP_ply_road)
        # gt_z = ICP_xh(target)

        pred_Tw,image,gt_image=refine(image=original_image,new_matrix=pred_Tw, maskpath = mask, depthpath = depth, gt_depth = gt_z)

        pred_t = pred_Tw[0:3, 3]
        t_bias = pred_t - gt_t
        add = compute_ADD(pred_Tw, gt_Tw, points)
        add_s = compute_ADD_S(pred_Tw, gt_Tw, points)
        angel = compute_angel(pred_Tw, gt_Tw)
        add_values.append(add)
        add_s_values.append(add_s)
        if add < 10:
            addcount += 1
        if add_s < 10:
            addscount += 1
        if add / max_diameter < 0.1:
            addcount_10 += 1
        if add_s / max_diameter < 0.1:
            addscount_10 += 1
        print(add, add_s)
        if angel<=5:
            refine_angle_num = refine_angle_num+1
        if add / max_diameter < 0.1 or add_s / max_diameter < 0.1:
            refine_juli_num = refine_juli_num+1
        else:
            filename = f"{wrong_folder}{count:06d}.png"  # i:06d会将i格式化为6位数，不足的前面补0
            cv2.imwrite(filename, image)

            #保存真值
            filename = f"{wrong_folder}{count:06d}_gt.png"  # i:06d会将i格式化为6位数，不足的前面补0
            cv2.imwrite(filename, gt_image)

        if add < 50 :
            refine_5cm_num = refine_5cm_num+1
        # count += 1
        print("X误差：", t_bias[0], "Y误差：",t_bias[1] , "Z误差：",t_bias[2] ,"角度误差：",angel)
        print(count,"ADD10:",refine_juli_num,"5度：",refine_angle_num,"5cm:",refine_5cm_num)
        print("ADD10_AR:", refine_juli_num/count, "5度_AR:", refine_angle_num/count,"5cm_AR:", refine_5cm_num/count)


    mean_add = np.mean(add_values)
    mean_add_s = np.mean(add_s_values)


