import torch
from gaussian_splatting.gaussian_renderer import render, nocs_render
import numpy as np
from argparse import ArgumentParser
import sys
from munch import munchify
from gaussian_splatting.scene.gaussian_model import GaussianModel
from torch import nn
import cv2
from gs2pose_utils.view_utils import *
from gs2pose_utils.config_utils import *
import os
import json

class LMRender:
    def __init__(self, config, ply_path, debug=True):
        self.debug = debug
        self.config = config
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
        # self.gaussians.training_setup(opt_params)
        bg_color = [0, 0, 0]
        self.background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    def run(self, viewpoint):
        # nocs_render

        render_pkg = render(
            viewpoint, self.gaussians, self.pipeline_params, self.background
        )
        # image, depth, opacity = (
        #     render_pkg["render"],
        #     render_pkg["depth"],
        #     render_pkg["opacity"],
        # )
        # img = image.to('cpu').permute(1, 2, 0).detach().numpy()
        # # 如果你的数据范围在 [0, 1]，你需要乘以 255 转换为 [0, 255] 范围
        # img = (img * 255).astype(np.uint8)
        # # 显示图像
        # cv2.imshow('Image', img)
        # cv2.waitKey(0)  # 等待按键

        return render_pkg

    def readCamerasFromSceneGt(self, path):
        cam_infos = []

        with open(os.path.join(path)) as json_file1:
            cam_extr = json.load(json_file1)

        for key, extr in cam_extr.items():

            R = np.array(extr[0]["cam_R_m2c"])
            T = np.array(extr[0]["cam_t_m2c"])
            R_n = R.reshape(3, 3)
            T_n = T
            T_CW = np_rt2mat(R_n,T)

            cam_infos.append(Camera(coarse_T=torch.tensor(T_CW, dtype=torch.float32, device='cuda'),  device=self.device, config = self.config))
        return cam_infos

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    parser.add_argument("--config", type=str, default="configs/mono/tum/000001.yaml")  # 配置文件
    parser.add_argument("--eval", action="store_true")  # 默认为eval模式

    args = parser.parse_args(sys.argv[1:])
    # 用`argparse`库中的`ArgumentParser`对象`parser`来解析从命令行传入的参数

    with open(args.config, "r") as yml:  # 打开一个文件，并使用`yaml`库的`safe_load()`函数加载该文件中的YAML配置。
        config = yaml.safe_load(yml)

    config = load_config(args.config)

    # plypath = "data/lm/models/obj_000006.ply"
    plypath = "output/000006/point_cloud/iteration_3600/point_cloud.ply"
    scene_gt_path = "data/lm/test/000006/scene_gt.json"
    rende_path = "test_render/lm/000006"          # "test_render/000001"

    lm_render = LMRender(config, plypath, debug=True)  # 配置主程序
    cameralist = lm_render.readCamerasFromSceneGt(scene_gt_path)
    idx = 0
    for viewpoint in cameralist:
        print(viewpoint.world_view_transform)
        render_pkg = lm_render.run(viewpoint)
        image, depth, opacity = (
            render_pkg["render"],
            render_pkg["depth"],
            render_pkg["opacity"],
        )
        image = (image.detach().permute(1, 2, 0).cpu().numpy()*255).astype(np.uint8)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        cv2.imshow("img", image)
        cv2.waitKey(50)
        cv2.imwrite(os.path.join(rende_path, '{0:06d}'.format(idx) + ".png"), image)
        idx += 1




