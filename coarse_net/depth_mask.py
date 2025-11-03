import cv2
import numpy as np
import os

# 定义深度图像和掩码图像的文件夹路径
depth_folder = "lmo_data/000001/depth/"
mask_folder = "data/lmo/mask_visib"
output_folder = "lmo_data/000001/depth_mask"

# 创建输出文件夹如果它不存在
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 获取所有深度图像的文件名
depth_files = os.listdir(depth_folder)

# 遍历所有深度图像文件
for filename in depth_files:
    if filename.endswith(".png"):  # 确保处理的是PNG文件
        # 构建完整的文件路径
        depth_path = os.path.join(depth_folder, filename)

        # 根据depth文件名构建对应的mask文件名
        # 假设mask文件名是在depth文件名的基础上加上了"_000000"后缀
        mask_filename = filename
        mask_path = os.path.join(mask_folder, mask_filename)

        # 检查mask文件是否存在
        if os.path.exists(mask_path):
            # 读取深度图像和掩码图像
            depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
            mask_gray = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)

            # 将掩码图像转换为灰度图像
            # mask_gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

            # 如果depth是3通道的，我们需要将其转换为单通道
            if depth.ndim == 3:
                depth = cv2.cvtColor(depth, cv2.COLOR_BGR2GRAY)

            # 将mask中为0的像素在depth中也设置为0
            depth[mask_gray == 0] = 0

            # 构建输出文件路径
            output_path = os.path.join(output_folder, filename)

            # 保存修改后的深度图像
            cv2.imwrite(output_path, depth)
        else:
            print(f"Mask file not found for {filename}")
