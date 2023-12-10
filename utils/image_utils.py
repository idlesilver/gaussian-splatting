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
from PIL import Image
from torchvision import transforms


def mse(img1, img2):
    return (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)


def psnr(img1, img2):
    mse = (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)
    return 20 * torch.log10(1.0 / torch.sqrt(mse))


# def load_images(image_path_list, resize: tuple = None):
#     """Load an image from a file path."""
#     imgs = []
#     for image_path in image_path_list:
#         img = Image.open(image_path)
#         if resize is not None:
#             img = img.resize(resize)
#         imgs.append(img)
#     return imgs


# def preprocess_image(imgs, patch_h, patch_w):
#     """Preprocess an image for feature extraction."""
#     preprocess = transforms.Compose([
#         transforms.Resize((patch_h*14, patch_w*14)),
#         # transforms.CenterCrop(224),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
#                              0.229, 0.224, 0.225]),
#     ])
#     imgs_t = torch.stack([preprocess(img) for img in imgs])
#     # Add batch dimension and move to device
#     return imgs_t
