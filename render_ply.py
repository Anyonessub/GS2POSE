import open3d as o3d
import json
import numpy as np


# 读取ply文件
def colored(point_cloud):
    # 获取点云数据
    points = np.asarray(point_cloud.points)
    print(points)
    # 归一化XYZ坐标到0-1的范围
    minp = points.min(axis=0)
    maxp = points.max(axis=0)
    points_normalized = (points - minp) / (maxp - minp)
    print(minp, maxp)

    # 将归一化后的XYZ坐标值转换为RGB颜色值
    colors = (points_normalized * 255).astype(np.uint8)

    # 将RGB颜色值存储在点云的颜色属性中
    point_cloud.colors = o3d.utility.Vector3dVector(colors / 255.0)

    # 需要时，从颜色值中提取RGB通道值并将其转换回原始的XYZ坐标值
    # 例如，假设要还原第一个点的XYZ坐标
    color = np.asarray(point_cloud.colors)[0] * 255
    xyz = color / 255.0 * (maxp - minp) + minp

    print("Original XYZ coordinate:", points[0])
    print("Restored XYZ coordinate:", xyz)


# # ply_path = "
# point_cloud = o3d.io.read_point_cloud("output/31c386b7-5/train/ours_7000/fuse.ply")
#
# # 创建一个窗口并显示点云
# o3d.visualization.draw_geometries([point_cloud])


# 读取camera.json文件
with open('data/lm/test/000006/scene_gt.json', 'r') as file:
    cam_extr = json.load(file)
with open('data/lm/test/000006/scene_camera.json', 'r') as file1:
    cam_intr = json.load(file1)

# 加载PLY点云数据
point_cloud = o3d.io.read_triangle_mesh("data/lm/models/obj_000006.ply")
# colored(point_cloud)

# 创建相机参数
intr = cam_intr["0"]["cam_K"]
height = 480
width = 640
focal_length_x = intr[0]
focal_length_y = intr[4]
cx = intr[2]
cy = intr[5]
print(height, width, focal_length_x, focal_length_y)
# 创建渲染窗口
vis = o3d.visualization.Visualizer()
vis.create_window(width=640, height=480)
# 设置渲染选项
render_option = vis.get_render_option()
# render_option.point_size = 4
render_option.background_color = [0, 0, 0]  # 将背景颜色设置为黑色
# render_option.light_on = False

camera_params = o3d.camera.PinholeCameraParameters()
camera_params.intrinsic.set_intrinsics(width, height, focal_length_x, focal_length_y,
                                       cx, cy)
i = 0
for key, extr in cam_extr.items():
    R = np.array(extr[0]["cam_R_m2c"])
    T = np.array(extr[0]["cam_t_m2c"])
    R_n = R.reshape(3, 3)
    # 设置相机外参矩阵
    extrinsic_matrix = np.eye(4)
    extrinsic_matrix[:3, :3] = R_n
    extrinsic_matrix[:3, 3] = T
    camera_params.extrinsic = extrinsic_matrix

    # 设置相机参数并渲染
    vis.add_geometry(point_cloud)
    vis.get_view_control().convert_from_pinhole_camera_parameters(camera_params, True)
    vis.poll_events()
    vis.update_renderer()
    vis.capture_screen_image(f"lm_rendered_test/000006_1/{i:06d}.png")
    # vis.capture_screen_image(f"render/rendered_image_{i:04d}.png")
    i = i + 1

    # 移除几何体以便下一次渲染
    vis.remove_geometry(point_cloud)

# 关闭窗口
vis.destroy_window()