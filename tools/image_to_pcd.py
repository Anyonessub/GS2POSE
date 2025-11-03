

import numpy as np
import cv2
import open3d as o3d

# 读取RGB图像和Depth图像
rgb_image_path = 'cut_result/000004/000000_000000.png'
depth_image_path = 'data/lm/test/000004/depth_mask/000000.png'

rgb_image = cv2.imread(rgb_image_path)
depth_image = cv2.imread(depth_image_path, cv2.IMREAD_UNCHANGED)  # 读取16位PNG深度图

# 将BGR转换为RGB
rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)

# 将深度图像转换为3D点云
if depth_image is not None:
    # 获取相机内参（这里假设相机内参已知）
    fx = 572.4114  # 相机的焦距
    fy = 573.57043
    cx = 325.2611  # 主点坐标
    cy = 242.04899

    # 构造3D点云
    depth_image_3d = depth_image.astype(np.float32) / 5000.0  # 将深度图像转换为3D坐标
    xx, yy = np.meshgrid(np.arange(depth_image.shape[1]), np.arange(depth_image.shape[0]))
    x = (xx - cx) * depth_image_3d / fx
    y = (yy - cy) * depth_image_3d / fy

    # 将深度图像的单通道转换为3通道
    points_3d = np.stack((x, y, depth_image_3d), axis=-1)

    # 创建Open3D点云
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_3d)
    pcd.colors = o3d.utility.Vector3dVector(rgb_image.reshape(-1, 3))

    # 保存点云到PLY文件
    o3d.io.write_point_cloud("output_point_cloud.ply", pcd)

else:
    print("Failed to load depth image.")













# import open3d as o3d
# import matplotlib.pyplot as plt
# import os
# import numpy as np
#
# # 定义RGB和深度图像的文件夹路径
# color_folder = "cut_result/000001/"
# depth_folder = "data/lm/test/000001/depth_mask/"
# output_folder = "data/lm/test/000001/ply/"
#
# # 创建输出文件夹如果它不存在
# if not os.path.exists(output_folder):
#     os.makedirs(output_folder)
#
# # 获取所有深度图像的文件名
# depth_files = os.listdir(depth_folder)
#
# # 遍历所有深度图像文件
# for filename in depth_files:
#     if filename.endswith(".png"):  # 确保处理的是PNG文件
#         # 构建完整的文件路径
#         color_filename = filename.replace(".png", "_000000.png")
#         color_path = os.path.join(color_folder, color_filename)
#         depth_path = os.path.join(depth_folder, filename)
#
#         # 检查RGB文件是否存在
#         if not os.path.exists(color_path):
#             print(f"RGB file not found for {filename}")
#             continue
#
#         # 读取RGB和深度图像
#         color_raw = o3d.io.read_image(color_path)
#         depth_raw = o3d.io.read_image(depth_path)
#
#         # 创建RGBD图像
#         rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
#             color_raw, depth_raw, convert_rgb_to_intensity=False)
#
#         # 设置相机内参
#         intrinsic = o3d.camera.PinholeCameraIntrinsic(
#             width=640, height=480, fx=572.4114, fy=573.57043, cx=325.2611, cy=242.04899)
#
#         # 生成点云
#         pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
#             rgbd_image, intrinsic)
#         # pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])  # 转换坐标系
#
#         # 保存点云
#         output_path = os.path.join(output_folder, filename.replace(".png", ".ply"))
#         o3d.io.write_point_cloud(output_path, pcd, write_ascii=True)  # 保存为PLY文件
#


