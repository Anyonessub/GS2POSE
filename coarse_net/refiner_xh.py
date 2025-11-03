import torch
import torch.nn.functional as F

from depth_mask import depth
from gaussian_splatting.gaussian_renderer import render
from gaussian_splatting.no_turn_gaussian_renderer import no_turn_render
import numpy as np
from argparse import ArgumentParser
import sys
from munch import munchify
from gaussian_splatting.scene.gaussian_model import GaussianModel
from torch import nn
import cv2
from view_utils import *
from config_utils import *
from loss_utils import *
from PIL import Image
import time


class Refiner:
    def __init__(self, config, ply_path, turn_view=True):
        self.turn_view = turn_view
        self.config = config
        self.tracking_itr_num = self.config["Training"]["tracking_itr_num"]
        model_params = munchify(config["model_params"])
        opt_params = munchify(config["opt_params"])
        pipeline_params = munchify(config["pipeline_params"])
        self.model_params, self.opt_params, self.pipeline_params = (
            model_params,
            opt_params,
            pipeline_params,
        )
        self.device = 'cuda'
        self.gaussians = GaussianModel(model_params.sh_degree, config=self.config)
        self.gaussians.load_ply(ply_path)
        # self.gaussians.fetchPly(ply_path)
        self.gaussians.init_lr(6.0)
        self.gaussians.training_setup(opt_params)
        # self.gaussians.training_setup(opt_params)
        bg_color = [0, 0, 0]
        self.background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        # 训练轮次
        self.first = 100
        self.second = 130
        self.third = 200
        self.final = 200
        self.depth_scale = 1.0
        self.use_depth = False
        self.depth=None

    def set_depth(self, depthpath, maskpath, gt_z, depth_scale=1.0):
        self.use_depth = True
        self.depth_scale = depth_scale
        mask_image = np.array(Image.open(maskpath))
        mask_image = torch.from_numpy(mask_image).to(
            dtype=torch.float32, device="cuda"
        )[None]
        gt_depth = np.array(Image.open(depthpath)) / self.depth_scale
        gt_depth = torch.from_numpy(gt_depth).to(
            dtype=torch.float32, device=self.device
        )[None]
        self.gt_z = gt_z
        self.depth = gt_depth
        self.mask = mask_image

    def reset_depth(self, viewpoint):
        valid_depths = self.depth * self.mask
        sum_depth = valid_depths.sum()
        num_valid_pixels = self.mask.sum()

        # 4. 计算平均深度
        # 注意：要避免除以零的情况
        if num_valid_pixels > 0:
            average_depth = sum_depth / num_valid_pixels
        else:
            average_depth = torch.tensor(0.0)  # 或者处理无有效区域的情况
        viewpoint.update_depth(average_depth)
        # viewpoint.update_depth(self.gt_z)

    def train(self, viewpoint, gt_image):

        opt_params = []
        if self.turn_view:
            opt_params.append(
                {
                    "params": [viewpoint.cam_rot_delta],
                    "lr": 0.01,  # self.config["Training"]["lr"]["cam_rot_delta"],
                    "name": "rot_{}".format(0),
                }
            )
            opt_params.append(
                {
                    "params": [viewpoint.cam_trans_delta],
                    "lr": 0.01,  # self.config["Training"]["lr"]["cam_trans_delta"],
                    "name": "trans_{}".format(0),
                }
            )
        else:
            opt_params.append(
                {
                    "params": [viewpoint.cam_rot_delta],
                    "lr": 0.01,  # self.config["Training"]["lr"]["cam_rot_delta"],
                    "name": "rot_{}".format(0),
                }
            )
            opt_params.append(
                {
                    "params": [viewpoint.cam_trans_delta],
                    "lr": 0.01,  # self.config["Training"]["lr"]["cam_trans_delta"],
                    "name": "trans_{}".format(0),
                }
            )
        opt_params.append(
            {
                "params": [viewpoint.exposure_a],
                "lr": 0.01,
                "name": "exposure_a_{}".format(0),
            }
        )
        opt_params.append(
            {
                "params": [viewpoint.exposure_b],
                "lr": 0.01,
                "name": "exposure_b_{}".format(0),
            }
        )

        pose_optimizer = torch.optim.Adam(opt_params)
        start_time = time.time()
        mini_loss=1000
        for tracking_itr in range(self.final):
            # print(tracking_itr)
            if (tracking_itr == self.first):
                self.turn_view = True
                for param_group in pose_optimizer.param_groups:
                    if param_group['params'][0] is viewpoint.cam_rot_delta:  # 确保找到正确的参数组
                        param_group['lr'] = 0.05  # 修改学习率为新的值，例如 0.001
                    if param_group['params'][0] is viewpoint.cam_trans_delta:  # 确保找到正确的参数组
                        param_group['lr'] = 0.05  # 修改学习率为新的值，例如 0.001

            if (self.turn_view):
                render_pkg = render(
                    viewpoint, self.gaussians, self.pipeline_params, self.background
                )
            else:
                render_pkg = no_turn_render(
                    viewpoint, self.gaussians, self.pipeline_params, self.background
                )
            image, depth, visibility_filter, radii, opacity = (
                render_pkg["render"],
                render_pkg["depth"],
                render_pkg["visibility_filter"],
                render_pkg["radii"],
                render_pkg["opacity"],
            )
            mask_shape = (1, 480, 640)
            opacity_mask = (opacity > 0.40).view(*mask_shape)
            gt_img = gt_image.to('cpu').permute(1, 2, 0).detach().numpy()

            # image = (torch.exp(viewpoint.exposure_a)) * image + viewpoint.exposure_b
            img = (image * opacity_mask).to('cpu').permute(1, 2, 0).detach().numpy()
            # 如果你的数据范围在 [0, 1]，你需要乘以 255 转换为 [0, 255] 范围
            img = (img * 255).astype(np.uint8)
            gt_img= (gt_img * 255).astype(np.uint8)

            # 显示图像
            cv2.imshow('Image', img)
            cv2.imshow('GTImage', gt_img)
            cv2.waitKey(10)  # 等待按键

            pose_optimizer.zero_grad()
            if (tracking_itr < self.first):
                if tracking_itr == 1 and self.use_depth:
                    self.reset_depth(viewpoint)
                loss_tracking = get_loss_tracking_rgb(
                    viewpoint, image, opacity, gt_image
                )
                # print(f"第一阶段 {loss_tracking}")
                loss_tracking.backward()
            elif (tracking_itr >= self.first and tracking_itr < self.second):
                if tracking_itr == self.first and self.use_depth:
                    self.reset_depth(viewpoint)
                # if tracking_itr == (self.final - 10) and self.use_depth:
                #     self.reset_depth(viewpoint)
                loss_tracking = get_loss_tracking_rgb_1(
                    viewpoint, image, opacity, self.mask ,gt_image
                )
                # print(f"第二阶段 {loss_tracking}")
                loss_tracking.backward()
            elif (tracking_itr >= self.second and tracking_itr < self.third):
                if tracking_itr == self.second and self.use_depth:
                    self.reset_depth(viewpoint)
                # elif (tracking_itr == (self.final - 10)):
                #     self.reset_depth(viewpoint)
                loss_tracking = get_loss_tracking_rgb_2(
                    viewpoint, image, opacity, self.mask ,gt_image
                )
                # print(f"第三阶段 {loss_tracking}")
                loss_tracking.backward()
            else:
                loss_tracking = get_loss_tracking_rgb_3(
                    viewpoint, image, opacity, gt_image
                )
                # print(f"第四阶段 {loss_tracking}")
                loss_tracking.backward()

            with torch.no_grad():
                if (tracking_itr >= self.second and tracking_itr < self.third):
                    self.gaussians.max_radii2D[visibility_filter] = torch.max(
                        self.gaussians.max_radii2D[visibility_filter],
                        radii[visibility_filter],
                    )
                    self.gaussians.optimizer.step()
                    self.gaussians.optimizer.zero_grad(set_to_none=True)
                    # self.gaussians.update_learning_rate(tracking_itr)
                pose_optimizer.step()
                converged = update_pose(viewpoint, self.turn_view)
                TP = viewpoint.get_T()
                # print("refine结果为:", np.linalg.inv(TP))

            if converged:
                break
        end_time = time.time()
        elapsed_time = end_time - start_time
        # print(f"程序运行时间: {elapsed_time:.2f}秒")
        return render_pkg, np.linalg.inv(TP), img, gt_img

    def train_rgbd(self, viewpoint, gt_image, gt_depth):

        opt_params = []
        if self.turn_view:
            opt_params.append(
                {
                    "params": [viewpoint.cam_rot_delta],
                    "lr": 0.01,  # self.config["Training"]["lr"]["cam_rot_delta"],
                    "name": "rot_{}".format(0),
                }
            )
            opt_params.append(
                {
                    "params": [viewpoint.cam_trans_delta],
                    "lr": 0.1,  # self.config["Training"]["lr"]["cam_trans_delta"],
                    "name": "trans_{}".format(0),
                }
            )
        else:
            opt_params.append(
                {
                    "params": [viewpoint.cam_rot_delta],
                    "lr": 0.01,  # self.config["Training"]["lr"]["cam_rot_delta"],
                    "name": "rot_{}".format(0),
                }
            )
            opt_params.append(
                {
                    "params": [viewpoint.cam_trans_delta],
                    "lr": 0.01,  # self.config["Training"]["lr"]["cam_trans_delta"],
                    "name": "trans_{}".format(0),
                }
            )
        opt_params.append(
            {
                "params": [viewpoint.exposure_a],
                "lr": 0.01,
                "name": "exposure_a_{}".format(0),
            }
        )
        opt_params.append(
            {
                "params": [viewpoint.exposure_b],
                "lr": 0.01,
                "name": "exposure_b_{}".format(0),
            }
        )

        pose_optimizer = torch.optim.Adam(opt_params)
        for tracking_itr in range(self.final):
            # print(tracking_itr)
            if (tracking_itr == self.first):
                self.turn_view = True
                for param_group in pose_optimizer.param_groups:
                    if param_group['params'][0] is viewpoint.cam_rot_delta:  # 确保找到正确的参数组
                        param_group['lr'] = 0.1  # 修改学习率为新的值，例如 0.001
                    if param_group['params'][0] is viewpoint.cam_trans_delta:  # 确保找到正确的参数组
                        param_group['lr'] = 0.01  # 修改学习率为新的值，例如 0.001

            if (self.turn_view):
                render_pkg = render(
                    viewpoint, self.gaussians, self.pipeline_params, self.background
                )
            else:
                render_pkg = no_turn_render(
                    viewpoint, self.gaussians, self.pipeline_params, self.background
                )
            image, depth, visibility_filter, radii, opacity = (
                render_pkg["render"],
                render_pkg["depth"],
                render_pkg["visibility_filter"],
                render_pkg["radii"],
                render_pkg["opacity"],
            )
            mask_shape = (1, 480, 640)
            opacity_mask = (opacity > 0.99).view(*mask_shape)
            gt_img = gt_image.to('cpu').permute(1, 2, 0).detach().numpy()
            img = (image * opacity_mask).to('cpu').permute(1, 2, 0).detach().numpy()
            # 如果你的数据范围在 [0, 1]，你需要乘以 255 转换为 [0, 255] 范围
            img = (img * 255).astype(np.uint8)
            gt_img= (gt_img * 255).astype(np.uint8)

            # 显示图像
            # cv2.imshow('Image', img)
            # cv2.imshow('GTImage', gt_img)
            # cv2.waitKey(20)  # 等待按键
            pose_optimizer.zero_grad()
            if (tracking_itr < self.first):
                loss_tracking = get_loss_tracking_rgb(
                    viewpoint, image, opacity, gt_image
                )
                loss_tracking.backward()
            elif (tracking_itr >= self.first and tracking_itr < self.second):
                loss_tracking = get_loss_tracking_rgb_1(
                    viewpoint, image, opacity, self.mask ,gt_image
                )
                loss_tracking.backward()
            else:
                loss_tracking = get_loss_tracking_rgbd(
                    image, gt_image, depth, gt_depth, opacity,viewpoint,self.mask
                )
                loss_tracking.backward()

            with torch.no_grad():
                pose_optimizer.step()
                converged = update_pose(viewpoint, self.turn_view)
                TP = viewpoint.get_T()
                # print("refine结果为:", np.linalg.inv(TP))

            if converged:
                break

        return render_pkg, viewpoint.get_T(),img,gt_img

    def run(self, coarse_T, gt_image, gt_depth=None):

        # gt_depth=self.depth
        viewpoint = Camera(coarse_T, self.config, device=self.device)
        if (gt_depth is not None):
            render_pkg, refine_T,image,gt_image = self.train_rgbd(viewpoint, gt_image,gt_depth=gt_depth)
            print("有深度数值")
        else:
            render_pkg, refine_T,image,gt_image = self.train(viewpoint, gt_image)
            print("无深度数值")
        return refine_T,image,gt_image


# if __name__ == "__main__":
def refine(image,new_matrix,maskpath,depthpath, gt_depth = None):
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    parser.add_argument("--config", type=str, default="configs/mono/tum/000001.yaml")  # 配置文件
    parser.add_argument("--eval", action="store_true")  # 默认为eval模式

    args = parser.parse_args(sys.argv[1:])
    # 用`argparse`库中的`ArgumentParser`对象`parser`来解析从命令行传入的参数

    with open(args.config, "r") as yml:  # 打开一个文件，并使用`yaml`库的`safe_load()`函数加载该文件中的YAML配置。
        config = yaml.safe_load(yml)

    config = load_config(args.config)

    plypath = "refine/000002/point_cloud/iteration_2000/point_cloud.ply"
    # plypath = "data/lm/models/obj_000001.ply"
    # imgpath = "cut_result/000001/000000_000000.png"
    # maskpath = "data/lm/test/000001/mask/000000_000000.png"
    # depthpath = "data/lm/test/000001/depth/000000.png"

    # gt_image = cv2.imread(imgpath)  # 放入待测图片

    # gt_image = cv2.cvtColor(gt_image, cv2.COLOR_RGB2BGR)
    # gt_image = zoom_image(gt_image,int(config["Dataset"]["Calibration"]["cx"]),int(config["Dataset"]["Calibration"]["cy"]),2)
    gt_image = image.astype(np.float32) / 255.0  # 转换为 [0, 1] 范围
    # 3. 将 NumPy 数组转换为 PyTorch 张量
    # OpenCV 读取的图像格式为 (height, width, channels)，需要转换为 (channels, height, width)
    gt_image = torch.from_numpy(gt_image).permute(2, 0, 1)  # 转换为 (C, H, W)

    # 4. 将张量移动到 CUDA 设备上（如果可用）
    if torch.cuda.is_available():
        gt_image = gt_image.cuda()

    # new_matrix = [[2.71103233e-01, 9.61791873e-01, 3.81820910e-02, -1.545007095e+02],
    #               [2.59678453e-01, -3.48845646e-02, -9.65064287e-01, -1.7479297e+02],
    #               [-9.26859796e-01, 2.71547198e-01, -2.59214461e-01, 1014.8770132],
    #               [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]]

    coarse_T = torch.tensor(new_matrix, dtype=torch.float32, device='cuda')

    refiner = Refiner(config, plypath, turn_view=False)  # 配置主程序

    refiner.set_depth(depthpath, maskpath, gt_depth)

    refine_T,image,gt_image = refiner.run(coarse_T, gt_image)  # 运行主程序

    return refine_T,image,gt_image

