# %%
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import torch
from torchvision import transforms
from PIL import Image
from matplotlib import pyplot as plt
import numpy as np
import os

# %%
device = 'cuda:0'
dinov2 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14').to(device)


# %%
def load_image(image_path_list):
    """Load an image from a file path."""
    imgs = []
    for image_path in image_path_list:
        imgs.append(Image.open(image_path).resize((1909//2, 1072//2)))
    return imgs


def preprocess_image(imgs, patch_h, patch_w):
    """Preprocess an image for feature extraction."""
    preprocess = transforms.Compose([
        transforms.Resize((patch_h*14, patch_w*14)),
        # transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                             0.229, 0.224, 0.225]),
    ])
    imgs_t = torch.stack([preprocess(img) for img in imgs])
    # Add batch dimension and move to device
    return imgs_t.to(device)


def extract_features(img_t):
    """Extract features from an image."""
    with torch.no_grad():
        features_dict = dinov2.forward_features(img_t)
        features = features_dict['x_norm_patchtokens']
    return features


def fea2rgb(feature):
    feature = ((feature - np.min(feature, axis=0)) /
               (np.max(feature, axis=0) - np.min(feature, axis=0))
               * 255
               ).astype('uint8')
    return feature


# %% Use the function
# image_path_list = r'data\tandt_db\db\playroom\images\DSC05572.jpg'
# image_path_list = [
#     r'D:\NextcloudRoot\research\gaussian-splatting\data\tandt_db\tandt\truck\images\000001.jpg'
# ]
source_path = r"D:\NextcloudRoot\research\gaussian-splatting\data\mydata\bottle"
image_path_list = [os.path.join(source_path, "images", image)
                   for image in ['0001.jpg',
                                 #  '0030.jpg',
                                 #  '0050.jpg',
                                 '0100.jpg',
                                 ]]
imgs = load_image(image_path_list)
img_h = imgs[0].size[0]
img_w = imgs[0].size[1]
patch_h = img_h // 14
patch_w = img_w // 14
print(f"n_shape: {(patch_h, patch_w)}")
imgs_t = preprocess_image(imgs, patch_h, patch_w)
features = extract_features(imgs_t)
print(f"patched semantic shape: {features.shape}")

# Assuming `features` is a 2D array where each row is a feature vector
features_np = features.cpu().detach().numpy()  # (bs, n_patch, n_feature)
bs, n_patch, n_feature = features.shape
features_np = features_np.reshape(-1, features_np.shape[-1])

# %% PCA
pca = PCA(n_components=3)
pca_features = pca.fit_transform(features_np)
print(f"PCA semantic feature shape: {pca_features.shape}")

# %% 
n_clusters = 3
instance = np.zeros_like(pca_features)
kmeans = KMeans(n_clusters, random_state=0,
                n_init="auto").fit(pca_features[..., 0].reshape(-1, 1))
cmap = pca.fit_transform(np.eye(n_clusters))*pca_features.max()

for i in range(n_clusters):
    if n_clusters > 3:
        instance[kmeans.labels_ == i] = cmap[i]
    else:
        instance[kmeans.labels_ == i] = kmeans.cluster_centers_[i]
print(f"Kmeans instance mask shape: {instance.shape}")
print(f"number of cluster center: {np.unique(instance, axis=0)}")

# %%
feature = np.concatenate([pca_features, instance], axis=0)
heatmap, instance_mask = np.array_split(fea2rgb(feature), 2)
heatmap = heatmap.reshape(bs, patch_h, patch_w, 3)
instance_mask = instance_mask.reshape(bs, patch_h, patch_w, 3)

# %% visualize
img_idx = 1
img = imgs[img_idx]
heatmap_img = Image.fromarray(heatmap[img_idx, ...]).resize(
    (img_h, img_w))
instance_mask_img = Image.fromarray(instance_mask[img_idx, ...]).resize(
    (img_h, img_w))

# %% show semantic heatmap
plt.figure()
plt.imshow(heatmap_img)
plt.axis('off')
plt.show()

# show instance mask
plt.figure()

plt.figure()
plt.imshow(instance_mask_img)
plt.axis('off')
plt.show()

# overlap heatmap
plt.figure()
plt.imshow(img)
plt.axis('off')
plt.show()

# overlap semantic
plt.figure()
plt.imshow(img)
plt.imshow(heatmap_img, alpha=0.8)
plt.axis('off')
plt.show()

# overlap instance mask
plt.figure()
plt.imshow(img)
plt.imshow(instance_mask_img, alpha=0.5)
plt.axis('off')
plt.show()

# %%
