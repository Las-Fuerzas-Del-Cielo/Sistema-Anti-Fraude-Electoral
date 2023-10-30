import math
from typing import *

import cv2
import numpy as np

from scipy import ndimage as nd
from skimage import img_as_bool
from skimage.transform import resize

from .misc import check_tensor


def scale_image(image: np.ndarray, mask: Optional[np.ndarray] = None,
                resolution: Union[int, Tuple[int, int]] = None, area: int = None, return_mask: bool = False) -> np.ndarray:

    dims = check_tensor(image, "h w c")
    check_tensor(mask, "h w", allow_none=True)

    assert 1 <= dims["c"] <= 3, "Invalid number of channels for input image"
    assert not (bool(resolution) and bool(area)), "Both scaling values cannot be set!"

    if not resolution and not area:  # no scaling parameter set: do nothing
        mask = np.ones_like(image[..., 0]).astype("bool") if mask is None else mask
        return (image, mask) if return_mask else image 

    if resolution:
        new_shape = resolution[::-1] if isinstance(resolution, Tuple) else (resolution, resolution)
    else:
        H, W, C = image.shape
        new_W = int(math.sqrt(area / (H / W)))
        new_H = int(area / new_W)
        new_shape = (new_W, new_H)

    if mask is None:
        scaled_image = cv2.resize(image, new_shape, interpolation=cv2.INTER_AREA)
        scaled_mask = np.ones_like(scaled_image[..., 0]).astype("bool")
        return (scaled_image, scaled_mask) if return_mask else scaled_image

    output_mask = img_as_bool(resize(mask, new_shape, preserve_range=True))

    indices = nd.distance_transform_edt(~mask.astype('bool'), return_distances=False, return_indices=True)
    filled_image = image[tuple(indices)]
    scaled_image = cv2.resize(filled_image, new_shape, interpolation=cv2.INTER_AREA)
    scaled_image[~output_mask] = 0

    return (scaled_image, output_mask) if return_mask else scaled_image


def tight_crop_image(image: np.ndarray, mask: np.ndarray, return_mask: bool = False):
    dims = check_tensor(image, "h w c")
    check_tensor(mask, "h w")
    assert 1 <= dims["c"] <= 3, "Number of dims is invalid!"

    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    row_min, row_max = np.where(rows)[0][[0, -1]]
    col_min, col_max = np.where(cols)[0][[0, -1]]

    image_crop = image[row_min:row_max, col_min:col_max]
    mask_crop = mask[row_min:row_max, col_min:col_max]

    return (image_crop, mask_crop) if return_mask else image_crop
