#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
from PIL import Image
import torchvision.transforms as T
from pathlib import Path
from scene.VGG import VGGEncoder, normalize_vgg

def render_set(model_path, name, iteration, views, gaussians, pipeline, background, style=None):

    if style:
        (style_img, style_name) = style
        render_path = os.path.join(model_path, name, style_name, "renders")
        gts_path = os.path.join(model_path, name, style_name, "gt")
        vgg_encoder = VGGEncoder().cuda()
        style_img_features = vgg_encoder(normalize_vgg(style_img))
    else:
        render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
        gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)

    override_color = None
    if style:
        tranfered_features = gaussians.style_transfer(
            gaussians.final_vgg_features.detach(), # point cloud features [N, C]
            style_img_features.relu3_1,
        )

        override_color = gaussians.decoder(tranfered_features) # [N, 3]

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        rendering = render(view, gaussians, pipeline, background, override_color=override_color)["render"]
        rendering = rendering.clamp(0, 1)
        gt = view.original_image[0:3, :, :]
        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))

def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, style_img_path, skip_train : bool, skip_test : bool):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        style = None
        if style_img_path:
            ckpt_path = os.path.join(dataset.model_path, "chkpnt/gaussians.pth")
            scene = Scene(dataset, gaussians, load_path=ckpt_path, shuffle=False, style_model=True)

            # read style image
            trans = T.Compose([T.Resize(size=(256,256)), T.ToTensor()])
            style_img = trans(Image.open(style_img_path)).cuda()[None, :3, :, :]
            style_name = Path(style_img_path).stem
            style = (style_img, style_name)
        else:
            scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if not skip_train:
             render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background, style)

        if not skip_test:
             render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background, style)


def render_sets_style_interpolate(dataset : ModelParams,  pipeline : PipelineParams, style_img_paths, view_id=0):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        ckpt_path = os.path.join(dataset.model_path, "chkpnt/gaussians.pth")
        scene = Scene(dataset, gaussians, load_path=ckpt_path, shuffle=False, style_model=True)

        # read 4 style images
        trans = T.Compose([T.Resize(size=(256,256)), T.ToTensor()])

        style_img0 = trans(Image.open(style_img_paths[0])).cuda()[None, :3, :, :]
        style_img1 = trans(Image.open(style_img_paths[1])).cuda()[None, :3, :, :]
        style_img2 = trans(Image.open(style_img_paths[2])).cuda()[None, :3, :, :]
        style_img3 = trans(Image.open(style_img_paths[3])).cuda()[None, :3, :, :]

        style_name0 = Path(style_img_paths[0]).stem
        style_name1 = Path(style_img_paths[1]).stem
        style_name2 = Path(style_img_paths[2]).stem
        style_name3 = Path(style_img_paths[3]).stem

        all_style_name = f'{style_name0}_{style_name1}_{style_name2}_{style_name3}'

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        render_path = os.path.join(dataset.model_path, "style_interpolation")
        makedirs(render_path, exist_ok=True)
        
        # get the style features
        vgg_encoder = VGGEncoder().cuda()
        style_img_features0 = vgg_encoder(normalize_vgg(style_img0))
        style_img_features1 = vgg_encoder(normalize_vgg(style_img1))
        style_img_features2 = vgg_encoder(normalize_vgg(style_img2))
        style_img_features3 = vgg_encoder(normalize_vgg(style_img3))

        # get the transfered features
        tranfered_features0 = gaussians.style_transfer(
            gaussians.final_vgg_features.detach(), 
            style_img_features0.relu3_1,
        )
        tranfered_features1 = gaussians.style_transfer(
            gaussians.final_vgg_features.detach(), 
            style_img_features1.relu3_1,
        )
        tranfered_features2 = gaussians.style_transfer(
            gaussians.final_vgg_features.detach(), 
            style_img_features2.relu3_1,
        )
        tranfered_features3 = gaussians.style_transfer(
            gaussians.final_vgg_features.detach(), 
            style_img_features3.relu3_1,
        )


        v = torch.linspace(0,1,steps=5)
        up_maps = []
        for i in range(5):
            up_maps.append(tranfered_features0 * v[i] + tranfered_features1 * v[4-i])
        down_maps = []
        for i in range(5):
            down_maps.append(tranfered_features2 * v[i] + tranfered_features3 * v[4-i])

        images = []
        w = torch.linspace(0,1,steps=4)
        for y in range(4):
            for x in range(5):
                tranfered_features_interpolated = up_maps[x] * w[y] + down_maps[x] * w[3-y]
                override_color = gaussians.decoder(tranfered_features_interpolated) # [N, 3]

                view = scene.getTrainCameras()[view_id]
                rendering = render(view, gaussians, pipeline, background, override_color=override_color)["render"]
                rendering = rendering.clamp(0, 1)

                rendering = torchvision.transforms.functional.resize(rendering, 300)

                images.append(rendering)

        torchvision.utils.save_image(images, fp=f'{render_path}/{all_style_name}.png', nrow=5, padding=0)


def render_sets_content_interpolate(dataset : ModelParams,  pipeline : PipelineParams, style_img_path, view_id=0):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        ckpt_path = os.path.join(dataset.model_path, "chkpnt/gaussians.pth")
        scene = Scene(dataset, gaussians, load_path=ckpt_path, shuffle=False, style_model=True)

        # read style features
        trans = T.Compose([T.Resize(size=(256,256)), T.ToTensor()])
        style_img = trans(Image.open(style_img_path)).cuda()[None, :3, :, :]
        style_name = Path(style_img_path).stem
        vgg_encoder = VGGEncoder().cuda()
        style_img_features = vgg_encoder(normalize_vgg(style_img))

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        render_path = os.path.join(dataset.model_path, "content_interpolation")
        makedirs(render_path, exist_ok=True)

        # get the transfered features
        tranfered_features = gaussians.style_transfer(
            gaussians.final_vgg_features.detach(), 
            style_img_features.relu3_1,
        )

        v = torch.linspace(0,1,steps=5)

        images = []
        for x in range(5):
            tranfered_features_interpolated = tranfered_features * v[x] + gaussians.final_vgg_features * v[4-x]
            override_color = gaussians.decoder(tranfered_features_interpolated)

            view = scene.getTrainCameras()[view_id]
            rendering = render(view, gaussians, pipeline, background, override_color=override_color)["render"]
            rendering = rendering.clamp(0, 1)

            rendering = torchvision.transforms.functional.resize(rendering, 300)

            images.append(rendering)

        torchvision.utils.save_image(images, fp=f'{render_path}/{style_name}.png', nrow=5, padding=0)


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--style", nargs='+', default='', type=str)
    parser.add_argument("--content_interpolate", action="store_true", default=False)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    if not args.style:
        render_sets(model.extract(args), args.iteration, pipeline.extract(args), None, args.skip_train, args.skip_test)
    if len(args.style) == 1: 
        if args.content_interpolate:
            render_sets_content_interpolate(model.extract(args), pipeline.extract(args), args.style[0])
        else:
            render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.style[0], args.skip_train, args.skip_test)
    elif len(args.style) == 4:
        render_sets_style_interpolate(model.extract(args), pipeline.extract(args), args.style)
    else:
        print("Invalid style argument, should provide 1 or 4 styles. 1 for style transfer, 4 for style interpolation.")