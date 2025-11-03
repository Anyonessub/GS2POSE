
from gaussian_splatting.gaussian_renderer import render
from argparse import ArgumentParser
import sys
from munch import munchify
from gaussian_splatting.scene.gaussian_model import GaussianModel
from PIL import Image
from gs2pose_utils.view_utils import *
from gs2pose_utils.config_utils import *
from gs2pose_utils.loss_utils import *
import os
import json
from random import randint

class Trainer:
    def __init__(self, config, debug=True):
        self.cam_list = []
        self.gt_images = []
        self.debug = debug
        self.config = config
        self.cameras_extent = 6.0
        self.tracking_itr_num = self.config["Training"]["tracking_itr_num"]
        self.gaussian_reset = self.config["Training"]["gaussian_reset"]
        self.gaussian_update_every = self.config["Training"]["gaussian_update_every"]
        self.gaussian_update_offset = self.config["Training"]["gaussian_update_offset"]
        self.gaussian_reset = self.config["Training"]["gaussian_reset"]
        self.gaussian_th = self.config["Training"]["gaussian_th"]
        self.gaussian_extent = (
            self.cameras_extent * self.config["Training"]["gaussian_extent"]
        )
        self.size_threshold = self.config["Training"]["size_threshold"]
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
        # self.gaussians.load_ply(ply_path)
        self.gaussians.init_lr(6.0)
        self.gaussians.full_training_setup(opt_params)
        bg_color = [0, 0, 0]
        self.background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    def train(self):

        loss_mapping = 0
        viewspace_point_tensor_acm = []
        visibility_filter_acm = []
        radii_acm = []
        n_touched_acm = []
        prune = False

        keyframes_opt = []
        for tracking_itr in range(2000):
            idx = randint(0, len(self.cam_list) - 1)
            viewpoint = self.cam_list[idx]
            render_pkg = render(
                viewpoint, self.gaussians, self.pipeline_params, self.background
            )
            (
                image,
                viewspace_point_tensor,
                visibility_filter,
                radii,
                depth,
                opacity,
                n_touched,
            ) = (
                render_pkg["render"],
                render_pkg["viewspace_points"],
                render_pkg["visibility_filter"],
                render_pkg["radii"],
                render_pkg["depth"],
                render_pkg["opacity"],
                render_pkg["n_touched"],
            )

            gt_img = viewpoint.gt_image.to('cpu').permute(1, 2, 0).detach().numpy()
            img = image.to('cpu').permute(1, 2, 0).detach().numpy()

            img = (img * 255).astype(np.uint8)
            gt_img= (gt_img * 255).astype(np.uint8)

            cv2.imshow('Image', img)
            cv2.imshow('GTImage', gt_img)
            cv2.waitKey(0)

            loss_mapping += get_loss_mapping_rgb(
                image, viewpoint
            )
            viewspace_point_tensor_acm.append(viewspace_point_tensor)
            visibility_filter_acm.append(visibility_filter)
            radii_acm.append(radii)
            n_touched_acm.append(n_touched)

            scaling = self.gaussians.get_scaling
            isotropic_loss = torch.abs(scaling - scaling.mean(dim=1).view(-1, 1))
            loss_mapping += 10 * isotropic_loss.mean()
            loss_mapping.backward()
            ## Deinsifying / Pruning Gaussians
            with torch.no_grad():
                for idx in range(len(viewspace_point_tensor_acm)):
                    self.gaussians.max_radii2D[visibility_filter_acm[idx]] = torch.max(
                        self.gaussians.max_radii2D[visibility_filter_acm[idx]],
                        radii_acm[idx][visibility_filter_acm[idx]],
                    )
                    self.gaussians.add_densification_stats(
                        viewspace_point_tensor_acm[idx], visibility_filter_acm[idx]
                    )

                update_gaussian = (
                        tracking_itr % self.gaussian_update_every
                        == self.gaussian_update_offset
                )
                if update_gaussian:
                    self.gaussians.densify_and_prune(
                        self.opt_params.densify_grad_threshold,
                        self.gaussian_th,
                        self.gaussian_extent,
                        self.size_threshold,
                    )
                if (tracking_itr % self.gaussian_reset) == 0 and (
                        not update_gaussian
                ):
                    self.gaussians.reset_opacity_nonvisible(visibility_filter_acm)
                self.gaussians.optimizer.step()
                self.gaussians.optimizer.zero_grad(set_to_none=True)
                self.gaussians.update_learning_rate(tracking_itr)

    def load_dataset(self, path):

        with open(os.path.join(path, "scene_gt.json")) as json_file1:
            cam_extr = json.load(json_file1)
        count = 0
        for key, extr in cam_extr.items():
            count += 1
            if count % 10 == 0:
                name = f"{int(key):06d}.png"
                gt_img_folder = os.path.join(path, "rgb")
                gt_image_path = os.path.join(gt_img_folder, os.path.basename(name))
                gt_image = np.array(Image.open(gt_image_path))
                gt_image = (
                    torch.from_numpy(gt_image / 255.0)
                    .clamp(0.0, 1.0)
                    .permute(2, 0, 1)
                    .to(device=self.device, dtype=torch.uint8)
                )

                R = np.array(extr[0]["cam_R_m2c"])
                T = np.array(extr[0]["cam_t_m2c"])
                R_n = R.reshape(3, 3)
                T_n = T
                T_CW = np_rt2mat(R_n,T)

                self.cam_list.append(Camera(torch.tensor(T_CW, dtype=torch.float32, device='cuda'),self.config, device=self.device, gt_image = gt_image))

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    parser.add_argument("--config", type=str)
    parser.add_argument("--eval", action="store_true")

    args = parser.parse_args(sys.argv[1:])


    with open(args.config, "r") as yml:
        config = yaml.safe_load(yml)

    config = load_config(args.config)

    dataset_path = "data/train/000001"


    trainer = Trainer(config, debug= True)
    trainer.load_dataset(dataset_path)

    trainer.train()

