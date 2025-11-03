

import numpy as np
from plyfile import PlyData


def read_ply_file(path):
    # 读取 PLY 文件
    plydata = PlyData.read(path)

    # 打印 PLY 文件的结构
    print(plydata.elements)



# 示例用法
ply_file_path = 'data1/lm/models/obj_000001.ply'  # 替换为你的 PLY 文件路径
plydata = read_ply_file(ply_file_path)

# def gaussian_background(image, mask):
#     # 1. 读取图像
#     img = cv2.imread(image)
#
#     # cv2.imshow("img", img)
#     # cv2.waitKey(0)
#
#     # 2. 转换为灰度图像并二值化
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     _, binary_mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
#
#     cv2.imshow("img", binary_mask)
#     cv2.waitKey(0)
#     # 3. 找到物体的轮廓
#     contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#
#     if len(contours) == 0:
#         raise ValueError("No object found in the image.")
#
#     # 选择最大的轮廓
#     largest_contour = max(contours, key=cv2.contourArea)
#
#     # 4. 计算物体的中心
#     M = cv2.moments(largest_contour)
#     if M["m00"] == 0:
#         raise ValueError("No valid moments found for the object.")
#
#     center_x = int(M["m10"] / M["m00"])
#     center_y = int(M["m01"] / M["m00"])
#     center = (center_x, center_y)
#
#     # 5. 计算物体的平均颜色
#     mask = np.zeros_like(img)
#     cv2.drawContours(mask, [largest_contour], -1, (255, 255, 255), -1)
#     mean_color = cv2.mean(img, mask=mask)[:3]  # 取前三个通道的平均值
#
#     # 6. 生成高斯渐变背景
#     height, width = img.shape[:2]
#     x = np.arange(0, width)
#     y = np.arange(0, height)
#     x, y = np.meshgrid(x, y)
#
#     # 高斯函数参数
#     sigma = 100  # 控制高斯分布的宽度
#     gaussian_background = np.exp(-((x - center_x) ** 2 + (y - center_y) ** 2) / (2 * sigma ** 2))
#
#     # 将高斯背景转换为BGR图像，并调整颜色
#     gaussian_background = (gaussian_background[:, :, np.newaxis] * np.array(mean_color)).astype(np.uint8)
#
#     # 7. 合成新图像
#     output_image = np.where(binary_mask[:, :, np.newaxis] == 255, img, gaussian_background)
#
#     return output_image, center, mean_color
#
#
# # 使用示例
# output_img, object_center, object_color = gaussian_background('result_image.jpg')
#
# # 保存或显示结果
# cv2.imwrite('output_image.png', output_img)
# print(f"Object Center: {object_center}, Average Color: {object_color}")