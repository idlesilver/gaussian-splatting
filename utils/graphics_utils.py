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
import math
import numpy as np
from typing import NamedTuple

class BasicPointCloud(NamedTuple):
    points : np.array
    colors : np.array
    normals : np.array

def geom_transform_points(points, transf_matrix):
    P, _ = points.shape
    ones = torch.ones(P, 1, dtype=points.dtype, device=points.device)
    points_hom = torch.cat([points, ones], dim=1)
    points_out = torch.matmul(points_hom, transf_matrix.unsqueeze(0))

    denom = points_out[..., 3:] + 0.0000001
    return (points_out[..., :3] / denom).squeeze(dim=0)

def getWorld2View(R, t):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0
    return np.float32(Rt)

def getWorld2View2(R, t, translate=np.array([.0, .0, .0]), scale=1.0):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0

    C2W = np.linalg.inv(Rt)
    cam_center = C2W[:3, 3]
    cam_center = (cam_center + translate) * scale
    C2W[:3, 3] = cam_center
    Rt = np.linalg.inv(C2W)
    return np.float32(Rt)

def getProjectionMatrix(znear, zfar, fovX, fovY):
    tanHalfFovY = math.tan((fovY / 2))
    tanHalfFovX = math.tan((fovX / 2))

    top = tanHalfFovY * znear
    bottom = -top
    right = tanHalfFovX * znear
    left = -right

    P = torch.zeros(4, 4)

    z_sign = 1.0
    # NOTE: 4x4 projection matrix.
#     GitHub Copilot: 这段代码中的`P`是一个4x4的投影矩阵，它的形式如下：
    # ```
    # P = | 2*znear/(right-left)  0                    (right+left)/(right-left)   0                          |
    #     | 0                     2*znear/(top-bottom) (top+bottom)/(top-bottom)   0                          |
    #     | 0                     0                    z_sign*zfar/(zfar-znear)    -(zfar*znear)/(zfar-znear) |
    #     | 0                     0                    z_sign                      0                          |
    # ```
    # 每一项的含义如下：
    # - `P[0, 0]`和`P[1, 1]`：这两项是x和y轴的缩放因子，它们决定了图像的宽高比。
    # - `P[0, 2]`和`P[1, 2]`：这两项是x和y轴的偏移因子，它们决定了图像的中心位置。
    # - `P[2, 2]`：这一项是z轴的缩放因子，它决定了近裁剪面和远裁剪面之间的深度范围。
    # - `P[2, 3]`：这一项是透视除法的因子，它用于将齐次坐标转换为笛卡尔坐标。
    # - `P[3, 2]`：这一项决定了z轴的方向，它可以是1.0（表示视点在z轴的正方向）或者-1.0（表示视点在z轴的负方向）。
    # 其他的元素都是0，因为它们不影响投影变换。
    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = (right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P

def fov2focal(fov, pixels):
    return pixels / (2 * math.tan(fov / 2))

def focal2fov(focal, pixels):
    return 2*math.atan(pixels/(2*focal))