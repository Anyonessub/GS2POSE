import open3d as o3d
import numpy as np

# 读取点云文件
point_cloud = o3d.io.read_point_cloud("refine/000013/point_cloud/iteration_5000/point_cloud.ply")  # 替换为你的点云文件路径

# 获取点云的 numpy 数组
points = np.asarray(point_cloud.points)

# 计算 x, y, z 的最大和最小值
x_min, x_max = np.min(points[:, 0]), np.max(points[:, 0])
y_min, y_max = np.min(points[:, 1]), np.max(points[:, 1])
z_min, z_max = np.min(points[:, 2]), np.max(points[:, 2])

# 打印结果
print(f"x 坐标的最小值: {x_min}, 最大值: {x_max}")
print(f"y 坐标的最小值: {y_min}, 最大值: {y_max}")
print(f"z 坐标的最小值: {z_min}, 最大值: {z_max}")
