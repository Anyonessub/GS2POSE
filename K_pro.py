import cv2
import numpy as np

# 相机内参矩阵（假设值，请根据实际情况进行替换）
cx = 325.2611,
cy = 242.04899,
fx = 572.4114,
fy = 573.57043,
height = 480
width = 640
camera_matrix = np.array([[572.4114, 0, 325.2611],[0, 573.57043, 242.04899],[0, 0, 1]], dtype=np.float32)

# 畸变系数（假设值，请根据实际情况进行替换）
dist_coeffs = np.array([0, 0, 0, 0, 0], dtype=np.float32)

# 读取输入图像
input_image = cv2.imread('data/lm/test/000001/rgb/000000.png')

# 获取图像尺寸
h, w = input_image.shape[:2]

# 计算新的相机内参矩阵（可以选择是否缩放）
new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, (w, h), 1, (w, h))

# 矫正图像
undistorted_image = cv2.undistort(input_image, camera_matrix, dist_coeffs, None, new_camera_matrix)

# 裁剪图像（如果需要）
x, y, w, h = roi
undistorted_image = undistorted_image[y:y+h, x:x+w]

# 显示结果
cv2.imshow('Undistorted Image', undistorted_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 保存结果
cv2.imwrite('undistorted_image.png', undistorted_image)
