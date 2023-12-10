import torch
from torchvision import transforms
from sklearn.decomposition import PCA
from torch import nn
import gc

N_COMPONENTS=3
dinov2 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14').cuda()
pca = PCA(n_components=N_COMPONENTS)

# def load_images(image_path_list, resize: tuple = None):
#     """Load an image from a file path."""
#     imgs = []
#     for image_path in image_path_list:
#         img = Image.open(image_path)
#         if resize is not None:
#             img = img.resize(resize)
#         imgs.append(img)
#     return imgs

def preprocess_image(img, patch_h, patch_w):
    """Preprocess an image for feature extraction."""
    preprocess = transforms.Compose([
        transforms.Resize((patch_h*14, patch_w*14)),
        # transforms.CenterCrop(224),
        # transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                             0.229, 0.224, 0.225]),
    ])
    img_t = preprocess(img)
    # Add batch dimension and move to device
    return img_t

def preprocess_images(imgs, patch_h, patch_w):
    imgs_t = torch.stack([preprocess_image(img,patch_h, patch_w) for img in imgs])
    # Add batch dimension and move to device
    return imgs_t

def extract_features(imgs):
    """Extract features from an image."""
    patch_h = imgs[0].shape[1] // 14
    patch_w = imgs[0].shape[2] // 14
    imgs_t = preprocess_images(imgs,patch_h,patch_w).cuda()
    with torch.no_grad():
        features_dict = dinov2.forward_features(imgs_t)
        features = features_dict['x_norm_patchtokens'].cpu().view(-1,1024, patch_h, patch_w).squeeze(0)
    # features = patch2pixel(features, imgs[0].shape[1:3])
    gc.collect()
    torch.cuda.empty_cache()
    return features

def patch2pixel(features, resolution):
    """Convert patch features to pixel features."""
    features = nn.functional.interpolate(
                    features.unsqueeze(0),
                    size=resolution,
                    mode="bilinear",
                    align_corners=False).squeeze(0)
    return features

def pca_features(features, pca):
    """Project features to PCA space."""
    features = torch.matmul(features, pca.components_.T)
    return features