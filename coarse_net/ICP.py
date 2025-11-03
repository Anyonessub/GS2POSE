import numpy as np
import open3d as o3d
from PIL import Image

def generate_point_cloud(rgb_image_path, depth_image_path):
    # 打开 RGB 图像
    rgb_image = Image.open(rgb_image_path).convert('RGB')
    # 打开深度图像并确保它是单通道
    depth_image = Image.open(depth_image_path).convert('L')  # 转换为灰度图像

    # 将 RGB 图像转换为 NumPy 数组
    rgb_array = np.array(rgb_image) / 255.0  # 归一化到 [0, 1]
    
    # 将深度图像转换为 NumPy 数组
    depth_array = np.array(depth_image)

    # 获取图像的宽度和高度
    height, width = depth_array.shape

    # 创建点云列表
    points = []
    colors = []

    # 遍历每个像素
    for y in range(height):
        for x in range(width):
            # 获取深度值
            z = depth_array[y, x] / 1000.0  # 假设深度值单位是毫米，转换为米

            # 仅处理有效的深度值
            if z > 0:
                # 计算 3D 点坐标
                points.append((x, y, z))
                colors.append((rgb_array[y, x][0], rgb_array[y, x][1], rgb_array[y, x][2]))

    # 转换为 NumPy 数组
    points = np.array(points)
    colors = np.array(colors)

    # 创建 Open3D 点云对象
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    return pcd

# 设置 RGB 和深度图像的路径
rgb_image_path = 'lmo_data/000001/depth/000356.png'  # 替换为实际 RGB 图像路径
depth_image_path = 'lmo_data/000001/result/000356.png'  # 替换为实际深度图像路径

# 生成点云
point_cloud = generate_point_cloud(rgb_image_path, depth_image_path)

# 可视化点云
o3d.visualization.draw_geometries([point_cloud])