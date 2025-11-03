import cv2
import os
import numpy as np
import open3d as o3d


def generate_and_save_point_cloud(rgb_image_path, depth_image_path, K, ply_output_path):
    # 读取RGB图像和深度图像
    rgb_image = cv2.imread(rgb_image_path)
    depth_image = cv2.imread(depth_image_path, cv2.IMREAD_UNCHANGED)

    # 获取图像大小
    height, width = depth_image.shape

    # 相机内参
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    # 创建点云列表
    points = []
    colors = []

    for v in range(height):
        for u in range(width):
            depth = depth_image[v, u]
            if depth == 0:  # 跳过无效的深度值
                continue

            # 计算相机坐标系下的3D坐标
            X_c = (u - cx) * depth / fx
            Y_c = (v - cy) * depth / fy
            Z_c = depth

            # 获取对应的RGB颜色
            color = rgb_image[v, u] / 255.0  # 转换到0-1范围

            points.append([X_c, Y_c, Z_c])
            colors.append(color)

    # 将点云转换为Open3D格式
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(np.array(points))
    point_cloud.colors = o3d.utility.Vector3dVector(np.array(colors))

    # 保存为PLY文件
    o3d.io.write_point_cloud(ply_output_path, point_cloud)

    # 可视化点云（可选）
    # o3d.visualization.draw_geometries([point_cloud])


def process_folder(rgb_folder, depth_folder, ply_folder, K):
    # 确保ply文件夹存在
    if not os.path.exists(ply_folder):
        os.makedirs(ply_folder)

    # 遍历depth文件夹下的所有图像
    for depth_filename in os.listdir(depth_folder):
        depth_image_path = os.path.join(depth_folder, depth_filename)
        color_filename = depth_filename.replace(".png", "_000000.png")
        rgb_image_path = os.path.join(rgb_folder, color_filename)  # 假设rgb和depth图像名字相同

        if os.path.isfile(depth_image_path) and os.path.isfile(rgb_image_path):
            # 构造PLY文件的保存路径
            ply_output_path = os.path.join(ply_folder, depth_filename.replace('.png', '.ply'))

            print(f"Processing RGB: {rgb_image_path} and Depth: {depth_image_path}")
            generate_and_save_point_cloud(rgb_image_path, depth_image_path, K, ply_output_path)


# 示例使用
K = np.array([[572.4114, 0, 325.2611], [0, 573.57043, 242.04899], [0, 0, 1]])  # 示例内参矩阵

# 文件夹路径（修改为你实际的文件夹路径）
rgb_folder = "cut_result/000002/"
depth_folder = "data/lm/test/000002/depth_mask/"
ply_folder = "data/lm/test/000002/ply/"

process_folder(rgb_folder, depth_folder, ply_folder, K)

