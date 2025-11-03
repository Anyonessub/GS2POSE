import open3d as o3d
import numpy as np

# 设置 alpha 值
alpha = 2000  # 你可以根据需要修改这个值

# 读取点云文件
point_cloud = o3d.io.read_point_cloud("data/lm/models/obj_000014.ply")

# 将点云转换为 numpy 数组
# points = np.asarray(point_cloud.points)
#
# print(points)
#
# # 筛选 X 轴小于等于 alpha 的点
# filtered_indices = points[:, 0] <= alpha
#
# # 更新点云数据
# point_cloud.points = o3d.utility.Vector3dVector(points[filtered_indices])
#
# # 如果点云有颜色，更新颜色数据
# if point_cloud.colors:
#     colors = np.asarray(point_cloud.colors)
#     point_cloud.colors = o3d.utility.Vector3dVector(colors[filtered_indices])
#
# # 如果点云有法向量，更新法向量数据
# if point_cloud.normals:
#     normals = np.asarray(point_cloud.normals)
#     point_cloud.normals = o3d.utility.Vector3dVector(normals[filtered_indices])

# 保存修改后的点云到同一个文件
o3d.io.write_point_cloud("filtered_point_cloud.ply", point_cloud)

# 展示修改后的点云
o3d.visualization.draw_geometries([point_cloud])