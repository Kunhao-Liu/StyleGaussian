import torch
import numpy as np
import os
import viser
import viser.transforms as tf
from collections import deque
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
from PIL import Image
from torchvision.transforms.functional import resize
import torchvision.transforms as T
from scene.VGG import VGGEncoder, normalize_vgg
from scene import Scene
from gaussian_renderer import render
from scene.cameras import Camera
from utils.general_utils import get_image_paths


class ViserViewer:
    def __init__(self, gaussians, pipeline, background, override_color, training_cams, wikiart_img_paths=None, viewer_port='8080'):
        self.gaussians = gaussians
        self.pipeline = pipeline
        self.background = background
        self.override_color = override_color
        self.fov = [training_cams[0].FoVx, training_cams[0].FoVy]

        self.port = viewer_port

        self.render_times = deque(maxlen=3)

        self.vgg_encoder = VGGEncoder().cuda()

        self.display_interpolation = False

        # Set up the server, init GUI elements
        self.server = viser.ViserServer(port=self.port)
        self.need_update = False

        with self.server.add_gui_folder("Rendering Settings"):
            self.reset_view_button = self.server.add_gui_button("Reset View")

            self.resolution_slider = self.server.add_gui_slider(
                "Resolution", min=384, max=2048, step=5, initial_value=1024
            )

            self.jpeg_quality_slider = self.server.add_gui_slider(
                "JPEG Quality", min=0, max=100, step=1, initial_value=80
            )

            self.training_view_slider = self.server.add_gui_slider(
                "Training View",
                min=0,
                max=len(training_cams) - 1,
                step=1,
                initial_value=0,
            )

            self.fps = self.server.add_gui_text("FPS", initial_value="-1", disabled=True)


        with self.server.add_gui_folder("Style Transfer"):
            self.style_img_path_text = self.server.add_gui_text(
                    "Style Image",
                    initial_value="",
                    hint="Path to style image",                
                )
            
            self.display_style_img = self.server.add_gui_checkbox("Display Style Image", initial_value=True)

            if wikiart_img_paths is not None:
                self.random_style_button = self.server.add_gui_button("Random Style")

        
        with self.server.add_gui_folder("Style Interpolation"):
            self.style_path_1 = self.server.add_gui_text(
                "Style 1",
                initial_value="",
                hint="Path to style image",
            )

            self.style_path_2 = self.server.add_gui_text(
                "Style 2",
                initial_value="",
                hint="Path to style image",
            )

            self.interpolation_ratio = self.server.add_gui_slider(
                "Interpolation Ratio",
                min=0,
                max=1,
                step=0.01,
                initial_value=0.5,
            )


        # Handle GUI events
        @self.resolution_slider.on_update
        def _(_):
            self.need_update = True

        @self.jpeg_quality_slider.on_update
        def _(_):
            self.need_update = True

        @self.reset_view_button.on_click
        def _(_):
            self.need_update = True
            for client in self.server.get_clients().values():
                client.camera.up_direction = tf.SO3(client.camera.wxyz) @ np.array(
                    [0.0, -1.0, 0.0]
                )

        @self.server.on_client_connect
        def _(client: viser.ClientHandle):
            @client.camera.on_update
            def _(_):
                self.need_update = True

        @self.training_view_slider.on_update
        def _(_):
            self.need_update = True
            for client in self.server.get_clients().values():
                target_camera = training_cams[self.training_view_slider.value]
                target_R = target_camera.R
                target_T = target_camera.T

                with client.atomic():
                    client.camera.wxyz = tf.SO3.from_matrix(target_R).wxyz
                    client.camera.position = -target_R @ target_T
                    self.fov = [target_camera.FoVx, target_camera.FoVy]
                    client.camera.up_direction = tf.SO3(client.camera.wxyz) @ np.array(
                        [0.0, -1.0, 0.0]
                    )

        @self.style_img_path_text.on_update
        def _(_):
            self.need_update = True
            self.display_interpolation = False
            style_img_path = self.style_img_path_text.value
            # read style image and extract features
            trans = T.Compose([T.Resize(size=(256,256)), T.ToTensor()])
            style_img = trans(Image.open(style_img_path)).cuda()[None, :3, :, :]
            style_img_features = self.vgg_encoder(normalize_vgg(style_img))
            self.style_img = resize(style_img, (128,128))

            # style transfer
            tranfered_features = self.gaussians.style_transfer(
                self.gaussians.final_vgg_features.detach(), # point cloud features [N, C]
                style_img_features.relu3_1,
            )
            self.override_color = self.gaussians.decoder(tranfered_features) # [N, 3]

        @self.display_style_img.on_update
        def _(_):
            self.need_update = True

        if wikiart_img_paths is not None:
            @self.random_style_button.on_click
            def _(_):
                self.need_update = True
                style_img_path = np.random.choice(wikiart_img_paths)
                self.style_img_path_text.value = style_img_path

        @self.style_path_1.on_update
        def _(_):
            style_interpolation()

        @self.style_path_2.on_update
        def _(_):
            style_interpolation()

        @self.interpolation_ratio.on_update
        def _(_):
            style_interpolation()

        def style_interpolation():
            if not self.style_path_1.value or not self.style_path_2.value:
                return
            
            self.need_update = True

            style_path_1 = self.style_path_1.value
            style_path_2 = self.style_path_2.value

            trans = T.Compose([T.Resize(size=(256,256)), T.ToTensor()])
            style_img0 = trans(Image.open(style_path_1)).cuda()[None, :3, :, :]
            style_img1 = trans(Image.open(style_path_2)).cuda()[None, :3, :, :]
        
            style_img_features0 = self.vgg_encoder(normalize_vgg(style_img0))
            style_img_features1 = self.vgg_encoder(normalize_vgg(style_img1))
            self.style_img0 = resize(style_img0, (128,128))
            self.style_img1 = resize(style_img1, (128,128))

            tranfered_features0 = gaussians.style_transfer(
                gaussians.final_vgg_features.detach(), 
                style_img_features0.relu3_1,
            )
            tranfered_features1 = gaussians.style_transfer(
                gaussians.final_vgg_features.detach(), 
                style_img_features1.relu3_1,
            )

            interpolated_features = (1-self.interpolation_ratio.value)*tranfered_features0 + self.interpolation_ratio.value*tranfered_features1

            self.override_color = self.gaussians.decoder(interpolated_features) # [N, 3]

            self.display_interpolation = True

    @torch.no_grad()
    def update(self):
        if self.need_update and self.override_color is not None:
            interval = None
            for client in self.server.get_clients().values():
                camera = client.camera
                R = tf.SO3(camera.wxyz).as_matrix()
                T = -R.T @ camera.position

                # get camera poses
                W = self.resolution_slider.value
                H = int(self.resolution_slider.value/camera.aspect)

                view = Camera(
                    colmap_id=None,
                    R=R,
                    T=T,
                    FoVx=self.fov[0],
                    FoVy=self.fov[1],
                    image=None, 
                    gt_alpha_mask=None,
                    image_name=None,
                    uid=None,
                )
                view.image_height = H
                view.image_width = W

                start_cuda = torch.cuda.Event(enable_timing=True)
                end_cuda = torch.cuda.Event(enable_timing=True)
                start_cuda.record()

                rendering = render(view, self.gaussians, self.pipeline, self.background, override_color=self.override_color)["render"]
                rendering = rendering.clamp(0, 1)
                if self.display_style_img.value:
                    if not self.display_interpolation and self.style_img is not None:
                        rendering[:, -128:, -128:] = self.style_img.squeeze(0)
                    elif self.style_path_1.value and self.style_path_2.value:
                        rendering[:, -128:, -128:] = self.style_img1.squeeze(0)
                        rendering[:, -128:, :128] = self.style_img0.squeeze(0)
                    
                end_cuda.record()
                torch.cuda.synchronize()
                interval = start_cuda.elapsed_time(end_cuda)/1000.

                out = rendering.permute(1,2,0).cpu().numpy().astype(np.float32)
                client.set_background_image(out, format="jpeg", jpeg_quality=self.jpeg_quality_slider.value)

            if interval:
                self.render_times.append(interval)
                self.fps.value = f"{1.0 / np.mean(self.render_times):.3g}"
            else:
                self.fps.value = "NA"


@torch.no_grad()
def run_viewer(dataset : ModelParams, pipeline : PipelineParams, wikiartdir, viewer_port):
    wikiart_img_paths = None
    if wikiartdir and os.path.exists(wikiartdir):
        print('Loading style images folder for random style transfer')
        wikiart_img_paths = get_image_paths(wikiartdir)

    # load trained gaussian model
    gaussians = GaussianModel(dataset.sh_degree)

    ckpt_path = os.path.join(dataset.model_path, "chkpnt/gaussians.pth")
    scene = Scene(dataset, gaussians, load_path=ckpt_path, shuffle=False, style_model=True)

    bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    # run viewer
    gui = ViserViewer(gaussians, pipeline, background, override_color=None, training_cams=scene.getTrainCameras(), wikiart_img_paths=wikiart_img_paths, viewer_port=viewer_port)
    while(True):
        gui.update()


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--style_folder", type=str, default="images")
    parser.add_argument("--viewer_port", type=str, default="8080")
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    run_viewer(model.extract(args), pipeline.extract(args), args.style_folder, viewer_port=args.viewer_port)