import math
from torch import nn
import numpy as np
from gs2pose_utils.li_utils import *

def fov2focal(fov, pixels):
    return pixels / (2 * math.tan(fov / 2))

def focal2fov(focal, pixels):
    return 2 * math.atan(pixels / (2 * focal))

def rt2mat(R, T):# 组合成变换矩阵
    mat = np.eye(4)
    mat[0:3, 0:3] = R.to('cpu').detach()
    mat[0:3, 3] = T.to('cpu').detach()
    return mat

def np_rt2mat(R, T):# 组合成变换矩阵
    mat = np.eye(4)
    mat[0:3, 0:3] = R
    mat[0:3, 3] = T
    return mat

def getWorld2View2(R, t, translate=torch.tensor([0.0, 0.0, 0.0]), scale=1.0):
    translate = translate.to(R.device)
    Rt = torch.zeros((4, 4), device=R.device)
    # Rt[:3, :3] = R.transpose()
    Rt[:3, :3] = R
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0

    C2W = torch.linalg.inv(Rt)
    cam_center = C2W[:3, 3]
    cam_center = (cam_center + translate) * scale
    C2W[:3, 3] = cam_center
    Rt = torch.linalg.inv(C2W)
    return Rt

def getProjectionMatrix2(znear, zfar, cx, cy, fx, fy, W, H):
    left = ((2 * cx - W) / W - 1.0) * W / 2.0
    right = ((2 * cx - W) / W + 1.0) * W / 2.0
    top = ((2 * cy - H) / H + 1.0) * H / 2.0
    bottom = ((2 * cy - H) / H - 1.0) * H / 2.0
    left = znear / fx * left
    right = znear / fx * right
    top = znear / fy * top
    bottom = znear / fy * bottom
    P = torch.zeros(4, 4)

    z_sign = 1.0

    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = (right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)

    return P

def update_pose(camera, turn_view):
    tau = torch.cat([camera.cam_trans_delta, camera.cam_rot_delta], axis=0)

    T_w2c = torch.eye(4, device=tau.device)
    T_w2c[0:3, 0:3] = camera.R
    T_w2c[0:3, 3] = camera.T

    if turn_view:
        # 修改为右乘扰动：
        new_w2c = T_w2c @ SE3_exp(tau)
    else:
        new_w2c = SE3_exp(tau) @ T_w2c # tau因该对应Tc1_c2 , Tc2_w = Tc2_c1 * Tc1_w(即T_w2c)

    new_R = new_w2c[0:3, 0:3]
    new_T = new_w2c[0:3, 3]
    converged_threshold = 1e-4
    converged = tau.norm() < converged_threshold # tau.norm()用于计算tau向量的范数
    camera.update_RT(new_R, new_T)

    camera.cam_rot_delta.data.fill_(0) # 将相机的旋转增量（cam_rot_delta）的数据填充为0
    camera.cam_trans_delta.data.fill_(0) # 将相机的位移增量（cam_trans_delta）的数据也填充为0
    return converged # 当更新量tau足够小时不再优化，跳出优化循环

class Camera(nn.Module):
    def __init__(
        self,
        colmap_id = None,
        R = None,
        T = None,
        FoVx = None,
        FoVy = None,
        image = None,
        gt_alpha_mask = None,
        image_name = None,
        uid = None,
        trans=np.array([0.0, 0.0, 0.0]),
        scale=1.0,
        coarse_T = None,
        device="cuda:0",
        gt_image = None,
        config = None
    ):
        super(Camera, self).__init__()
        self.gt_image = gt_image
        self.device = device
        self.image_name = image_name
        self.trans = trans
        self.scale = scale
        if coarse_T is not None:
            self.R = coarse_T[:3, :3]
            self.T = coarse_T[:3, 3]
        else:
            self.R = torch.from_numpy(R).to(self.device)
            self.T = torch.from_numpy(T).to(self.device)
        if config is not None:
            self.fx = config["Dataset"]["Calibration"]["fx"]
            self.fy = config["Dataset"]["Calibration"]["fy"]
            self.cx = config["Dataset"]["Calibration"]["cx"]
            self.cy = config["Dataset"]["Calibration"]["cy"]
            self.image_height = config["Dataset"]["Calibration"]["height"]
            self.image_width = config["Dataset"]["Calibration"]["width"]
            self.fovx = focal2fov(self.fx, self.image_width)  # 计算水平市场角
            self.fovy = focal2fov(self.fy, self.image_height)
        else:
            self.fx = 572.4114
            self.fy = 573.57043
            self.cx = 325.2611
            self.cy = 242.04899
            self.image_height = 242.04899 * 2
            self.image_width = 325.2611 * 2
            self.uid = uid
            self.colmap_id = colmap_id
            self.fovx= FoVx
            self.fovy = FoVy
            self.data_device = device

        self.projection_matrix = self.projection_matrix().to('cuda')

        self.cam_rot_delta = nn.Parameter(
            torch.zeros(3, requires_grad=True, device=device)
        )
        self.cam_trans_delta = nn.Parameter(
            torch.zeros(3, requires_grad=True, device=device)
        )

        self.exposure_a = nn.Parameter(
            torch.tensor([0.0], requires_grad=True, device=device)
        )
        self.exposure_b = nn.Parameter(
            torch.tensor([0.0], requires_grad=True, device=device)
        )
        if image is not None:
            self.original_image = image.clamp(0.0, 1.0).to(self.data_device)
            self.gt_mask = image.clamp(0.0, 1.0).to(self.data_device)
            self.image_width = self.original_image.shape[2]
            self.image_height = self.original_image.shape[1]

            if gt_alpha_mask is not None:
                self.original_image *= gt_alpha_mask.to(self.data_device)
            else:
                self.original_image *= torch.ones((1, self.image_height, self.image_width), device=self.data_device)

    def update_RT(self, R, t):
        self.R = R.to(device=self.device)
        self.T = t.to(device=self.device)

    def update_depth(self, depth):
        self.T[2] = depth

    def update_xy(self, x, y):
        self.T[0] = y
        self.T[1] = x

    def get_T(self):
        return rt2mat(self.R, self.T)
    # def get_final_T(self):
    #     self.T[2] /= 1.05
    #     return rt2mat(self.R, self.T)
    def get_R(self):
        mat = np.eye(3)
        mat = self.R.to('cpu').detach()
        return mat
    @property
    def world_view_transform(self):
        return getWorld2View2(self.R, self.T).transpose(0, 1).requires_grad_(True)# 注意这里也进行了转置

    @property
    def full_proj_transform(self):
        return (
            self.world_view_transform.unsqueeze(0).bmm(
                self.projection_matrix.unsqueeze(0)
            ) # bmm表示批量矩阵乘法（batch matrix multiplication），输入是两个三维张量，形状分别为 (B, N, M) 和 (B, M, P)
        ).squeeze(0) # 注意这里的world_view_transform与projection_matrix都是转置后的，所以矩阵乘法是左右颠倒的

    def projection_matrix(self): #计算透视投影矩阵，矩阵将三维空间中的点投射到二维平面上
        projection_matrix = getProjectionMatrix2(
            znear=0.01, zfar=100.0, fx=self.fx, fy=self.fy, cx=self.cx, cy=self.cy, W=self.image_width, H=self.image_height
        ).transpose(0, 1).requires_grad_(True) # 注意这里进行了转置
        return projection_matrix
    @property
    def camera_center(self):
        return self.world_view_transform.inverse()[3, :3]


def zoom_image(image, u, v, n):
    # 获取原图像的尺寸
    h, w = image.shape[:2]

    # 计算新的尺寸
    new_w, new_h = int(w * n), int(h * n)

    # 创建新的图像
    new_image = np.zeros((new_h, new_w, 3), dtype=image.dtype)

    # 放大图像
    for y in range(new_h):
        for x in range(new_w):
            # 计算原图像中的对应像素
            orig_x = (x - (u * n)) / n + u
            orig_y = (y - (v * n)) / n + v

            # 确保索引在原图像范围内
            if 0 <= orig_x < w and 0 <= orig_y < h:
                new_image[y, x] = image[int(orig_y), int(orig_x)]

    # 将放大后的图像裁剪到640x480
    start_x = max(0, u * n - 320)
    start_y = max(0, v * n - 240)

    cropped_image = new_image[start_y:start_y + 480, start_x:start_x + 640]

    return cropped_image