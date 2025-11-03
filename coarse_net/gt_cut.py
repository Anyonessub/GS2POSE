import cv2
import os
import numpy as np

# 定义文件夹路径
mask_folder = 'data/lmo/mask_visib'  # mask文件夹路径
rgb_folder = 'data/lmo/image'    # rgb文件夹路径
result_folder = 'data/lmo/image2'  # 结果文件夹路径

# 创建结果文件夹（如果不存在的话）
if not os.path.exists(result_folder):
    os.makedirs(result_folder)

# 获取mask文件夹中所有文件的名称
mask_files = sorted(os.listdir(mask_folder))

for mask_file in mask_files:
    rgb_file_name = mask_file.split('_')[0] + '.png'
    # 构建mask和RGB图像的完整路径
    mask_path = os.path.join(mask_folder, mask_file)
    rgb_path = os.path.join(rgb_folder, rgb_file_name)

    # 读取mask和RGB图像
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    rgb_image = cv2.imread(rgb_path)

    # 检查图像是否成功读取
    if mask is None or rgb_image is None:
        print(f"Error reading {mask_file}. Skipping...")
        continue

    # 创建一个三通道的mask
    mask_3channel = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

    # 应用mask
    result = cv2.bitwise_and(rgb_image, mask_3channel)

    # 保存结果
    result_path = os.path.join(result_folder, mask_file)
    cv2.imwrite(result_path, result)

    print(f"Processed {mask_file}")

print("All images processed.")