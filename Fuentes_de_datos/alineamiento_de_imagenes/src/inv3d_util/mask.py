import math
from typing import *

import numpy as np

from skimage import img_as_bool
from skimage.transform import resize

from .misc import check_tensor

def scale_mask(mask: np.ndarray, resolution: Union[int, Tuple[int, int]] = None, area: int = None) -> np.ndarray:
    check_tensor(mask, "h w")

    assert not (bool(resolution) and bool(area)), "Both scaling values cannot be set!"

    if not resolution and not area:  # no scaling parameter set: do nothing
        return mask

    if resolution:
        new_shape = resolution[::-1] if isinstance(resolution, Tuple) else (resolution, resolution)
    else:
        H, W = mask.shape
        new_W = int(math.sqrt(area / (H / W)))
        new_H = int(area / new_W)
        new_shape = (new_W, new_H)

    output_mask = img_as_bool(resize(mask, new_shape))

    return output_mask
    

def tight_crop_mask(mask: np.ndarray, return_mask: bool = False):
    check_tensor(mask, "h w")

    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    row_min, row_max = np.where(rows)[0][[0, -1]]
    col_min, col_max = np.where(cols)[0][[0, -1]]

    mask_crop = mask[row_min:row_max, col_min:col_max]

    return mask_crop
