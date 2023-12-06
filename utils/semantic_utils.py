import torch

device = 'cuda:0'
dinov2 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14').to(device)


def extract_features(img_t):
    """Extract features from an image."""
    with torch.no_grad():
        features_dict = dinov2.forward_features(img_t)
        features = features_dict['x_norm_patchtokens']
    return features

