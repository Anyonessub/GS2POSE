import numpy as np
import open3d as o3d
from scipy.spatial import KDTree, cKDTree

def translate(points, t):
    return points + t

def rotate(points, R):
    return np.dot(points, R.T)

def select_points_based_view(rotation_matrix, source):
    points = np.asarray(source.points)

    transformation_matrix = np.eye(4)  # 创建一个4x4的单位矩阵
    transformation_matrix[:3, :3] = rotation_matrix  # 将旋转矩阵放入左上角

    # 将点云点坐标转换到相机坐标系
    ones = np.ones((points.shape[0], 1))  # 增加一列1以便进行齐次坐标变换
    points_homogeneous = np.hstack((points, ones))  # (N, 4) 形状
    transformed_points = transformation_matrix @ points_homogeneous.T  # (4, N) 形状
    transformed_points = transformed_points.T[:, :3]  # 取前3列，转换为 (N, 3) 形状

    # 筛选面向相机的点
    # 这里假设相机在z轴的正方向，z坐标应该为正
    # valid_points = transformed_points[transformed_points[:, 2] < 0]

    # 处理面向相机的点，保留xy坐标近的点中z坐标最小的点
    unique_xy = {}
    for point in transformed_points:
        xy_key = (round(point[0]//4), round(point[1]//4))  # 保留xy坐标到小数点后6位
        if xy_key not in unique_xy:
            unique_xy[xy_key] = point
        else:
            # 如果已经存在，则比较z坐标，保留z坐标最小的点
            if point[2] < unique_xy[xy_key][2]:
                unique_xy[xy_key] = point

    # 将结果转换为numpy数组
    filtered_points = np.array(list(unique_xy.values()))

    # 创建新的点云
    # 创建新的点云

    transformer_pcd = o3d.geometry.PointCloud()
    transformer_pcd.points = o3d.utility.Vector3dVector(transformed_points)
    # 设置颜色为红色
    transformer_pcd.colors = o3d.utility.Vector3dVector(np.tile([0.0, 0.3, 0.6], (transformed_points.shape[0], 1)))  # Re


    filtered_pcd = o3d.geometry.PointCloud()
    filtered_pcd.points = o3d.utility.Vector3dVector(filtered_points)
    # 设置颜色为蓝色
    filtered_pcd.colors = o3d.utility.Vector3dVector(np.tile([1, 0, 0], (filtered_points.shape[0], 1)))  # Blue



    # 可视化两个点云
    # o3d.visualization.draw_geometries([transformer_pcd],
    #                                   window_name="ICP Registration - Matching Points",
    #                                   width=800, height=600, left=50, top=50)

    return filtered_pcd


def select_best_matching_points(source_points, target_points):
    """
    从源点云中选择与目标点云相同数量的点，使用KDTree进行匹配，确保每个点只能选择一次
    """
    num_target_points = target_points.shape[0]
    selected_indices = set()  # 存储已选择的源点索引
    matched_source_points = []

    # 使用KDTree进行最近邻搜索
    tree = cKDTree(source_points)

    for target_point in target_points:
        # 找到目标点云中当前点的最近邻
        distance, index = tree.query(target_point, k=1)

        # 确保未选择过的点
        if index not in selected_indices:
            matched_source_points.append(source_points[index])
            selected_indices.add(index)

    matched_source_points = np.array(matched_source_points)

    return matched_source_points, target_points

def nearest_neighbor(src, dst):
    # tree = KDTree(dst)
    tree = cKDTree(dst)
    distances, indices = tree.query(src, k=1)
    return indices

def compute_transform(src, dst):
    src_mean = np.mean(src, axis=0)
    dst_mean = np.mean(dst, axis=0)
    src_centered = src - src_mean
    dst_centered = dst - dst_mean
    H = np.dot(src_centered.T, dst_centered)
    U, _, Vt = np.linalg.svd(H)
    R = np.dot(Vt.T, U.T)
    t = dst_mean - np.dot(R, src_mean)
    return R, t

def filter_point_cloud(point_cloud):
    # 应用体素网格滤波器
    voxel_size = 3# 体素大小，可以根据需要调整  3
    downsampled = point_cloud.voxel_down_sample(voxel_size)

    # 应用统计滤波器
    cl, ind = downsampled.remove_statistical_outlier(nb_neighbors=200, std_ratio=0.5)
    filtered_point_cloud = downsampled.select_by_index(ind)

    return filtered_point_cloud

def compute_z_transformation(source, target):
    # 计算源点云和目标点云之间的Z值变换
    # 只考虑Z轴的平移
    return np.mean(target[:, 2]) - np.mean(source[:, 2])


def icp_update_z(source, target):

    z_translation = compute_z_transformation(source, target)

    return z_translation

def ICP_gt_show(target ,gt_matrix):
    source = o3d.io.read_point_cloud("data/lm/models/obj_0000.ply")
    source.paint_uniform_color([1, 0, 0])
    source.transform(gt_matrix)
    target.paint_uniform_color([0, 0, 1])
    # Visualize only the matched points from source and target
    # Visualize the original source in a separate window
    # o3d.visualization.draw_geometries([source, target],
    #                                   window_name="ICP Registration - Original Source",
    #                                   width=800, height=600, left=100, top=100)


def ICP_xh(target, rotate_matrix ,transform_matrix, get_xy = False):
    # source = o3d.io.read_point_cloud("output/000014/point_cloud/iteration_3600/point_cloud.ply")
    source = o3d.io.read_point_cloud("data/lm/models/obj_000015.ply")
    # source = o3d.io.read_point_cloud("filtered_point_cloud.ply")
    # 取得xy坐标备份
    t_ori = transform_matrix[:3,3]
    t_xy = t_ori.copy()

    # 对souce进行颜色变换
    colors = np.asarray(source.colors)
    # BGR转换为RGB，只需要交换第0和第2列
    colors[:, [0, 2]] = colors[:, [2, 0]]
    # 将转换后的颜色重新赋值给点云
    source.colors = o3d.utility.Vector3dVector(colors)

    # 将source旋转到粗估网络的角度上，并且筛选下面向相机一侧的点
    source = select_points_based_view(rotate_matrix, source)
    # 可视化筛选结果
    # o3d.visualization.draw_geometries([source],
    #                                   window_name="ICP Registration - Original Source",
    #                                   width=800, height=600, left=100, top=100)

    # 对待匹配的目标rgbd图片进行滤波
    source = filter_point_cloud(source)
    # 获取两个点云的坐标集合
    source_points = np.array(source.points)
    target = filter_point_cloud(target)
    target_points = np.array(target.points)

    if get_xy == False: # 精估前只更新z(因为xy不准)
        # 获得第一次z值预测
        z_est = icp_update_z(source_points, target_points)
        # print("\nEstimated z:", z_est)
        return z_est
    # 精估第一阶段后，将xyz一起更新，并且进入邻近点匹配
    t_xy[2] = 0
    coarse_points = translate(source_points, t_xy)
    z_est = icp_update_z(coarse_points, target_points)
    print("\nEstimated z:", z_est)
    t_xyz = np.array([0,0,z_est])
    coarse_points = translate(coarse_points, t_xyz)

    # 可视化当前结果：
    coarse = o3d.geometry.PointCloud()
    coarse.points = o3d.utility.Vector3dVector(coarse_points)
    coarse.paint_uniform_color([0, 0, 1])
    target.paint_uniform_color([1, 0, 0])

    # o3d.visualization.draw_geometries([target, coarse],
    #                                   window_name="ICP Registration - Matching Points",
    #                                   width=800, height=600, left=50, top=50)

    # o3d.visualization.draw_geometries([target],
    #                                   window_name="ICP Registration - Matching Points",
    #                                   width=800, height=600, left=50, top=50)

    return z_est

    # # matched_source_points = icp_points[indices]
    # matched_source_points,_ = select_best_matching_points(icp_points,target_points)
    # corresponding_target_points = target_points
    #
    # t_est1 = icp_update_z(matched_source_points, corresponding_target_points)
    # matched_source_points = translate(matched_source_points, t_est1)
    #
    # # print("\nEstimated Rotation Matrix:", R_est)
    # print("\nEstimated Translation Vector:", t_est1)
    #
    # # Create point clouds for the matching points
    # matching_source = o3d.geometry.PointCloud()
    # matching_target = o3d.geometry.PointCloud()
    #
    # # Assign only the matched points
    # matching_source.points = o3d.utility.Vector3dVector(matched_source_points)
    # matching_source.paint_uniform_color([1, 0, 0])  # Red for transformed matched source points
    #
    # matching_target.points = o3d.utility.Vector3dVector(corresponding_target_points)
    # matching_target.paint_uniform_color([0, 0, 1])  # Blue for corresponding target points
    #
    # # Visualize only the matched points from source and target
    # # Visualize the original source in a separate window
    # o3d.visualization.draw_geometries([matching_source],
    #                                   window_name="ICP Registration - Original Source",
    #                                   width=800, height=600, left=100, top=100)
    #
    # o3d.visualization.draw_geometries([matching_source, matching_target],
    #                                   window_name="ICP Registration - Matching Points",
    #                                   width=800, height=600, left=50, top=50)

    # return t_est[2]# + t_est1[2]  #注意换回原来的KD要该会来







