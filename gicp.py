import open3d as o3d
import numpy as np

def load_point_cloud(file_path):
    pcd = o3d.io.read_point_cloud(file_path)
    return pcd

def apply_gicp(source, target):
    # 初始化GICP
    threshold = 0.02  # 设定距离阈值
    reg_gicp = o3d.pipelines.registration.registration_generalized_icp(
        source, target, threshold,
        np.eye(4),
        o3d.pipelines.registration.TransformationEstimationForGeneralizedICP()
    )
    return reg_gicp.transformation

def main():
    # 加载源点云和目标点云
    source_pcd = load_point_cloud("data/lm/models/obj_000002.ply")
    target_pcd = load_point_cloud("data/lm/test/000002/ply/000000.ply")

    # 执行GICP配准
    transformation = apply_gicp(source_pcd, target_pcd)

    print(transformation)

    # 应用变换到源点云
    source_pcd.transform(transformation)

    # 可视化配准结果
    o3d.visualization.draw_geometries([source_pcd, target_pcd])

if __name__ == "__main__":
    main()