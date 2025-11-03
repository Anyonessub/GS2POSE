import torch
import torch.nn.functional as F

from gaussian_splatting.gaussian_renderer import render, nocs_render
from gaussian_splatting.no_turn_gaussian_renderer import no_turn_render
import numpy as np
from argparse import ArgumentParser
import sys
from munch import munchify
from gaussian_splatting.scene.gaussian_model import GaussianModel
from torch import nn
import cv2
from gs2pose_utils.view_utils import *
from gs2pose_utils.config_utils import *
from gs2pose_utils.loss_utils import *
from PIL import Image
import time
import open3d as o3d
from ICP import ICP_xh

class Refiner:
    def __init__(self, config ,ply_path, gt_T= None, turn_view=True, gui = False):
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
        self.first = 45 # 45
        self.second = 140 #140
        self.third = 160 # 160
        self.final = 175 # 175
        self.depth_scale = 1.0
        self.use_depth = False
        self.depth=None
        self.gt_T = gt_T
        self.use_gui = gui

    def set_depth(self, depthpath, maskpath, gt_z, d_ply_path ,depth_scale=1.0):
        self.use_depth = True
        self.depth_scale = depth_scale
        #对标签数据与深度数据进行读取
        mask_image = np.array(Image.open(maskpath))
        cv2.imshow("mask_image",mask_image)
        cv2.waitKey(30)
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
        self.target = o3d.io.read_point_cloud(d_ply_path)

    def set_xy_center(self, viewpoint):

        binary_image = (self.mask > 127).float()


        white_pixels = torch.nonzero(binary_image[0], as_tuple=False)

        centroid = white_pixels.float().mean(dim=0)

        z = viewpoint.T[2]
        print(centroid, z)
        x = -1 * (centroid[0]- viewpoint.cx) * z / viewpoint.fx
        y = -1 * (centroid[1]- viewpoint.cy) * z / viewpoint.fy

        print(x,y)

        viewpoint.update_xy(y, x)

    def reset_depth(self, viewpoint):

        viewpoint.update_depth(self.gt_z)

    def icp_depth(self, viewpoint, get_xy = False):
        new_z = ICP_xh(self.target, viewpoint.get_R() ,viewpoint.get_T(), get_xy)
        # ICP_gt_show(self.target, self.gt_T)
        viewpoint.T[2] = new_z
        return viewpoint

    def train(self, viewpoint, gt_image, gt_depth, gt_mask):

        opt_params = []
        opt_params.append(
            {
                "params": [viewpoint.cam_rot_delta],
                "lr": 0.001,  # self.config["Training"]["lr"]["cam_rot_delta"],
                "name": "rot_{}".format(0),
            }
        )
        opt_params.append(
            {
                "params": [viewpoint.cam_trans_delta],
                "lr": 0.005,  # self.config["Training"]["lr"]["cam_trans_delta"], # 10
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
        # start_time = time.time()

        for tracking_itr in range(self.final):
            # print(tracking_itr)
            if tracking_itr == self.first:
                self.turn_view = True
                for param_group in pose_optimizer.param_groups:
                    if param_group['params'][0] is viewpoint.cam_rot_delta:
                        param_group['lr'] = 0.05
                    if param_group['params'][0] is viewpoint.cam_trans_delta:
                        param_group['lr'] = 0.01
            elif tracking_itr == self.second:
                self.turn_view = False
                for param_group in pose_optimizer.param_groups:
                    if param_group['params'][0] is viewpoint.cam_rot_delta:
                        param_group['lr'] = 0.001
                    if param_group['params'][0] is viewpoint.cam_trans_delta:
                        param_group['lr'] = 0.01
            elif tracking_itr == self.third:
                self.turn_view = True
                for param_group in pose_optimizer.param_groups:
                    if param_group['params'][0] is viewpoint.cam_rot_delta:
                        param_group['lr'] = 0.01
                    if param_group['params'][0] is viewpoint.cam_trans_delta:
                        param_group['lr'] = 0.01

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

            if self.use_gui == True:
                mask_shape = (1, 480, 640)
                opacity_mask = (opacity > 0.40).view(*mask_shape)
                gt_img = gt_image.to('cpu').permute(1, 2, 0).detach().numpy()

                # image = (torch.exp(viewpoint.exposure_a)) * image + viewpoint.exposure_b
                img = (image * opacity_mask).to('cpu').permute(1, 2, 0).detach().numpy()

                img = (img * 255).astype(np.uint8)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                gt_img= (gt_img * 255).astype(np.uint8)
                gt_img = cv2.cvtColor(gt_img, cv2.COLOR_BGR2RGB)

                alpha = 0.9
                beta = 0.8

                blended = cv2.addWeighted(img, alpha, gt_img, beta, 0)

                cv2.imshow('Blended Image', blended)
                cv2.waitKey(1)

            pose_optimizer.zero_grad()
            if (tracking_itr < self.first):
                if tracking_itr == 0 and self.use_depth:
                    self.icp_depth(viewpoint)
                # loss_tracking = get_loss_tracking_rgb_2(
                #     viewpoint, image, opacity, self.mask, gt_image
                # )
                loss_tracking = get_loss_tracking_rgbd(
                    viewpoint, image, opacity, gt_image, depth,gt_depth,gt_mask
                )
                loss_tracking.backward()
            elif (tracking_itr >= self.first and tracking_itr < self.second):
                if (tracking_itr == self.first or tracking_itr == self.second - 2) and self.use_depth:
                    self.icp_depth(viewpoint, get_xy=True)
                # loss_tracking = get_loss_tracking_rgb_2(
                #     viewpoint, image, opacity, self.mask ,gt_image
                # )
                loss_tracking = get_loss_tracking_rgbd(
                    viewpoint, image, opacity, gt_image, depth,gt_depth,gt_mask
                )
                loss_tracking.backward()
            elif (tracking_itr >= self.second and tracking_itr < self.third):
                self.icp_depth(viewpoint, get_xy=False)
                # loss_tracking = get_loss_tracking_rgb_1(
                #     viewpoint, image, opacity, self.mask ,gt_image
                # )
                loss_tracking = get_loss_tracking_rgbd(
                    viewpoint, image, opacity, gt_image, depth,gt_depth,gt_mask
                )

                loss_tracking.backward()
            else:
                if tracking_itr == self.second and self.use_depth:
                    self.icp_depth(viewpoint)
                # loss_tracking = get_loss_tracking_rgb_1(
                #     viewpoint, image, opacity, self.mask ,gt_image
                # )
                loss_tracking = get_loss_tracking_rgbd(
                    viewpoint, image, opacity, gt_image, depth,gt_depth,gt_mask
                )
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

            if converged:
                break
        # end_time = time.time()
        # elapsed_time = end_time - start_time
        return render_pkg, TP


    def run(self, coarse_T, gt_image, gt_depth,gt_mask):

        # gt_depth=self.depth
        viewpoint = Camera(coarse_T = coarse_T, config = self.config, device = self.device)
        if (gt_depth is not None):
            render_pkg, refine_T = self.train(viewpoint, gt_image, gt_depth, gt_mask)
            print("有深度数值")
        else:
            render_pkg, refine_T= self.train(viewpoint, gt_image)
            print("无深度数值")
        return refine_T


# if __name__ == "__main__":
def refine(image,new_matrix,maskpath,depthpath,d_ply_path, gt_depth = None, gt_T = None, index = None, gui = False):
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    parser.add_argument("--config", type=str, default="configs/mono/tum/000001.yaml")
    parser.add_argument("--eval", action="store_true")

    args = parser.parse_args(sys.argv[1:])

    with open(args.config, "r") as yml:
        config = yaml.safe_load(yml)

    config = load_config(args.config)

    plypath = f"output/{index:06d}/point_cloud/iteration_3600/point_cloud.ply"

    gt_image = image.astype(np.float32) / 255.0
    gt_image = torch.from_numpy(gt_image).permute(2, 0, 1)

    gt_depth = cv2.imread(depthpath, cv2.IMREAD_UNCHANGED)
    gt_depth = gt_depth.astype(np.float32)
    gt_depth = torch.tensor(gt_depth).unsqueeze(0)
    gt_mask = cv2.imread(maskpath, cv2.IMREAD_GRAYSCALE)
    _, gt_mask = cv2.threshold(gt_mask, 127, 255, cv2.THRESH_BINARY)
    gt_mask = torch.tensor(gt_mask)

    if torch.cuda.is_available():
        gt_image = gt_image.cuda()
        gt_depth = gt_depth.cuda()
        gt_mask = gt_mask.cuda()


    coarse_T = torch.tensor(new_matrix, dtype=torch.float32, device='cuda')

    refiner = Refiner(config, plypath, gt_T = gt_T ,turn_view=False, gui = gui)

    refiner.set_depth(depthpath, maskpath, gt_depth, d_ply_path)

    refine_T = refiner.run(coarse_T, gt_image,gt_depth,gt_mask)

    return refine_T


