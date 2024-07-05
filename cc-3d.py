import torch

import cupy as cp
from cucim.skimage import measure as cucim_measure

def get_connected_components(img, connectivity=None):

    img_cupy = cp.asarray(img)
    labeled_img, num_features = cucim_measure.label(img_cupy, connectivity=connectivity, return_num=True)
    labeled_img_torch = torch.as_tensor(labeled_img, device=img.device)

    return labeled_img_torch, num_features
