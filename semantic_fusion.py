# %%
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import torch
from scene import Scene, GaussianModel
from arguments import ModelParams
from utils.camera_utils import Camera
import os
source_path = r"D:\NextcloudRoot\research\gaussian-splatting\data\mydata\bottle"

# %% setup train context


class ModelParams():
    def __init__(self):
        self.sh_degree = 3
        self.source_path = source_path
        self.model_path = ""
        self.images = "images"
        self.resolution = -1
        self.white_background = False
        self.data_device = "cuda"
        self.eval = False


dataset = ModelParams()
gaussians = GaussianModel(3)
scene = Scene(dataset, gaussians, shuffle=False)

# %% get sence info
origin_xyzs = gaussians.get_xyz
cams = scene.train_cameras[1]
h, w = cams[0].image_height, cams[0].image_width

# %% show the distribution of x of cloud points.
# NOTE: there are OOD points in the cloud points.
# they should be randomly initialized for training.
# NOTE: the missing part might caused by the mirror reflection.
plt.figure()
plt.plot(sorted(origin_xyzs[:, 0].cpu().detach().numpy()))
plt.figure()
plt.hist(sorted(origin_xyzs[:, 0].cpu().detach().numpy()))

# %% project the point back to image space
c_idx = 0
p_idx = [0]
# xyzs = origin_xyzs[p_idx] # use only selected points
xyzs = origin_xyzs  # use all points
cam: Camera = cams[c_idx]
xyzw = torch.cat([xyzs, torch.ones_like(xyzs[..., 0:1])], dim=-1)
xyzk = cam.full_proj_transform.T @ xyzw.T
xyz = xyzk / xyzk[-1, :]
# xyzk, xyz

# %% exclude the OOD points and find their color
xy = xyz[0:2, :].cpu().detach().numpy()
color = gaussians._color.cpu().detach().numpy()
mask = (np.abs(xy[0]) < 1) & (np.abs(xy[1]) < 1)
xy = xy[:, mask]
c = color[mask, :]
c = (c+c.min()) / (c.max()-c.min())

# %% render the projected point splatting
x = ((xy[0] + 1.0) * h - 1.0) * 0.5
y = -((xy[1] + 1.0) * w - 1.0) * 0.5
plt.scatter(x, y, c=c)

# %%
img = Image.open(os.path.join(source_path, "images",
                 f"{int(cam.image_name):04d}.jpg"))
img
# %%
