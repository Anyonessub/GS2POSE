import cv2
import numpy as np
import open3d as o3d
import OpenEXR

from utils.utils_metrics import compute_ADD, compute_ADD_S, compute_angel
import os
import json


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
    # #0002
    # minp = np.asarray([-41.2807, -23.5249, -41.7435])
    # maxp = np.asarray([41.4265, 23.5842, 41.6361])
    #0003
    # minp = np.asarray([-31.9354, -31.7139, -14.5107])
    # maxp = np.asarray([31.9436, 31.6627, 14.3547])
    #0004
    # minp = np.asarray([-26.2880, -27.5745, -19.4907])
    # maxp = np.asarray([26.5324, 27.6799, 19.2426])
    #0005
    # minp = np.asarray([ -50.3958, -90.8979, -96.867])
    # maxp = np.asarray([50.3958, 90.8979, 96.867])
    #0006
    # minp = np.asarray([-33.5054, -63.8165, -58.7283])
    # maxp = np.asarray([33.5053, 63.8165, 58.7283])
    #0008
    # minp = np.asarray([-44.0788, -15.1495, -42.0531])
    # maxp = np.asarray([44.0067, 15.1201, 39.8918])
    #0009
    minp = np.asarray([-52.2146,-38.7038, -42.8485])
    maxp = np.asarray([52.2146, 38.7038, 42.8485])
    #0010
    # minp = np.asarray([-75.0923,-53.5375,  -34.6207])
    # maxp = np.asarray([ 75.0923, 53.5375, 34.6207])
    #0011
    # minp = np.asarray([-18.3605,-38.9330,  -86.4079])
    # maxp = np.asarray([ 18.3606, 38.9330, 86.4079])
    #0012
    # minp = np.asarray([-19.5269, -20.9922, -17.3149])
    # maxp = np.asarray([19.5734, 21.0395, 17.4438])
    #0013
    # minp = np.asarray([-49.4115, -23.3701, -27.4184])
    # maxp = np.asarray([49.6738, 23.4330, 27.1159])
    #0014
    # minp = np.asarray([-101.5730, -58.8763,  -106.5580])
    # maxp = np.asarray([101.5730, 58.8762, 106.5580])
     #0015
    # minp = np.asarray([-18.4732, -28.7324, -36.5970])
    # maxp = np.asarray([18.7158, 28.7718, 35.4058])

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

    source_path = "0001/test/scene_gt.json"
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
    ply_path = "0001/test/obj_000001.ply"
    point_cloud = o3d.io.read_point_cloud(ply_path)

    # 从点云中提取3D点坐标
    points = np.asarray(point_cloud.points)
    # distances = [np.linalg.norm(p1 - p2) for p1, p2 in combinations(points, 2)]

    # 找到最大距离
    max_diameter = 102  # max(distances)
    # print(max_diameter)

    # 指定文件夹路径
    folder_path = "./0001/result/"

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
        if count == 50:
            break

    # 计算ADD与ADD-S值并求平均
    add_values = []
    add_s_values = []
    t_bias = []
    count = 0
    angle_num=0
    juli_num=0
    for pred_Tw, gt_Tw in zip(pred_Tw2c, gt_Tw2c):
        print(pred_Tw)   #输入精估网络
        # print(gt_Tw)
        pred_t = pred_Tw[0:3, 3]
        gt_t = gt_Tw[0:3, 3]
        t_bias.append(pred_t - gt_t)
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
        print(add, add_s, angel)
        if angel<10:
            angle_num = angle_num+1
        if add < 100 and add_s < 100:
            juli_num = juli_num+1
        count += 1
        print(count,juli_num,angle_num)

    mean_add = np.mean(add_values)
    mean_add_s = np.mean(add_s_values)

    # print("平均ADD值：", mean_add)
    # print("平均ADD-S值：", mean_add_s)
    # print("ADD AR:", addcount)
    # print("ADD-S AR:", addscount)
    # print("ADD-10 AR:", addcount_10)
    # print("ADD-S-10 AR:", addscount_10)
