from pathlib import Path
from typing import *

import scipy
import warnings
import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange
from scipy.interpolate import RegularGridInterpolator
from scipy.interpolate import griddata
from torch import Tensor
from torchvision.transforms.functional import rotate
from scipy.interpolate import InterpolatedUnivariateSpline


from .load import load_npz
from .misc import check_tensor


def create_identity_map(resolution: Union[int, Tuple], with_margin: bool = False):
    if isinstance(resolution, int):
        resolution = (resolution, resolution)

    margin_0 = 0.5 / resolution[0] if with_margin else 0
    margin_1 = 0.5 / resolution[1] if with_margin else 0
    return np.mgrid[margin_0:1-margin_0:complex(0, resolution[0]), margin_1:1-margin_1:complex(0, resolution[1])].transpose(1, 2, 0)


def apply_map(image: np.ndarray, bm: np.ndarray, resolution: Union[None, int, Tuple[int, int]] = None):
    check_tensor(image, "h w c")
    check_tensor(bm, "h w c", c=2)

    if resolution is not None:
        bm = scale_map(bm, resolution)

    input_dtype = image.dtype
    img = rearrange(image, 'h w c -> 1 c h w')
    img = torch.from_numpy(img).double()

    bm = torch.from_numpy(bm).unsqueeze(0).double()
    bm = (bm * 2) - 1
    bm = torch.roll(bm, shifts=1, dims=-1)

    res = F.grid_sample(input=img, grid=bm, align_corners=True)
    res = rearrange(res[0], 'c h w -> h w c')
    res = res.numpy().astype(input_dtype)
    return res


def apply_map_torch(image: Tensor, bm: Tensor, resolution: Union[None, int, Tuple[int, int]] = None):
    check_tensor(image, "n c h w")
    check_tensor(bm, "n 2 h w")

    if resolution is not None:
        bm = scale_map_torch(bm, resolution)

    input_dtype = image.dtype
    image = image.double()

    bm = rearrange(bm, "n c h w -> n h w c").double()
    bm = (bm * 2) - 1
    bm = torch.roll(bm, shifts=1, dims=-1)

    res = F.grid_sample(input=image, grid=bm, align_corners=True)
    res = res.type(input_dtype)

    return res


def invert_map(input_map: np.ndarray, extrapolate: bool = True):
    check_tensor(input_map, "h h c", c=2)

    resolution, _, _ = input_map.shape
    mask = ~np.any(np.isnan(input_map), axis=-1)

    points = input_map[mask]
    values = np.array(np.nonzero(mask)).transpose((1, 0)).astype(float) / resolution
    id_map = create_identity_map(resolution, with_margin=True)

    flow_grid = griddata(points=points, values=values, xi=(id_map[..., 0], id_map[..., 1]), method='linear')

    if extrapolate and np.isnan(flow_grid).any():
        extrapolation = griddata(points=points, values=values, xi=(id_map[..., 0], id_map[..., 1]), method='nearest')
        flow_grid = np.where(np.isnan(flow_grid), extrapolation, flow_grid)

    return flow_grid

def invert_map_torchlike(input_map: Tensor, extrapolate: bool = True) -> Tensor:
    check_tensor(input_map, "n c h h", c=2)

    device = input_map.device
    result = []

    # TODO parallelize!
    for forward_map in input_map:
        forward_map = rearrange(forward_map, 'c h w -> h w c').detach().cpu().numpy()
        backward_map = invert_map(forward_map, extrapolate=extrapolate)
        backward_map = rearrange(backward_map, 'h w c -> c h w') 
        backward_map = torch.from_numpy(backward_map).to(device)
        
        result.append(backward_map)

    return torch.stack(result)


def scale_map(input_map: np.ndarray, resolution: Union[int, Tuple[int, int]]):
    try:
        resolution = (int(resolution), int(resolution))
    except (ValueError, TypeError):
        resolution = tuple(int(v) for v in resolution)

    H, W, C = input_map.shape
    if H == resolution[0] and W == resolution[1]:
        return input_map.copy()

    if np.any(np.isnan(input_map)):
        print("WARNING: scaling maps containing nan values will result in unsteady borders!")
    
    x = np.linspace(0, 1, W)
    y = np.linspace(0, 1, H)
    xi = create_identity_map(resolution, with_margin=False).reshape(-1, 2) # TODO might be important, or not?

    interp = RegularGridInterpolator((y, x), input_map, method="linear") # TODO keep
    return interp(xi).reshape(*resolution, C)


def scale_map_torch(input_map: Tensor, resolution: Union[int, Tuple[int, int]]):
    check_tensor(input_map, "n 2 h w")

    try:
        resolution = (int(resolution), int(resolution))
    except (ValueError, TypeError):
        resolution = tuple(int(v) for v in resolution)

    B, C, H, W = input_map.shape
    if H == resolution[0] and W == resolution[1]:
        return input_map

    if torch.any(torch.isnan(input_map)):
        print("WARNING: scaling maps containing nan values will result in unsteady borders!")

    return F.interpolate(input_map, size=resolution, mode='bilinear', align_corners=True)


def tight_crop_map(input_map: np.ndarray):
    check_tensor(input_map, "h w 2")

    input_map -= np.nanmin(input_map, axis=(0, 1), keepdims=True)
    input_map /= np.nanmax(input_map, axis=(0, 1), keepdims=True)
    return input_map

def tight_crop_map_torch(input_map: Tensor) -> Tensor:
    check_tensor(input_map, "n 2 h w")

    for idx, tensor in enumerate(input_map):
        valid_mask = ~torch.any(torch.isnan(tensor), dim=0)

        valid_data = tensor[:, valid_mask]
        min_values = valid_data.amin(dim=1).unsqueeze(dim=-1).unsqueeze(dim=-1)
        max_values = valid_data.amax(dim=1).unsqueeze(dim=-1).unsqueeze(dim=-1)

        input_map[idx] -= min_values
        input_map[idx] /= max_values

    return input_map


def rotate_map_torch(input_map: Tensor, angles: Tensor) -> Tensor:
    dims = check_tensor(input_map, 'n 2 h h')
    check_tensor(angles, 'n', n=dims['n'])

    input_map = scale_map_torch(input_map, dims['h']*10)  # upsample map to avoid rotation artefacts

    rot_map = torch.stack([rotate(data, float(angle), expand=True, fill=float("nan")) 
                           for data, angle in zip(input_map, angles)])

    # crop nan values and resize map
    maps = []
    for tensor in rot_map:
        mask = ~torch.any(torch.isnan(tensor), dim=0)

        rows = torch.any(mask, dim=-1)
        cols = torch.any(mask, dim=-2)
        row_min, row_max = torch.nonzero(rows, as_tuple=False)[[0, -1]]
        col_min, col_max = torch.nonzero(cols, as_tuple=False)[[0, -1]]

        crop_map = tensor[:, row_min:row_max, col_min:col_max]
        crop_map = crop_map.unsqueeze(0)
        crop_map = scale_map_torch(crop_map, dims['h'])  # downsample again

        maps.append(crop_map)

    return torch.concat(maps)


def extrapolate_torchlike(input_map: torch.Tensor):
    dims = check_tensor(input_map, 'n 2 h h')
    resolution = dims['h']
    device = input_map.device

    input_map = input_map.detach().cpu().numpy()
    input_map = rearrange(input_map, "n c h w -> n h w c")

    def interp1d(y):
        x = np.linspace(0, 1, resolution)
        m = ~np.isnan(y)

        return y if m.sum() <= 1 else InterpolatedUnivariateSpline(x[m], y[m], k=1)(x)

    results = []

    for tensor in input_map:
        while np.sum(np.isnan(tensor)) > 0:
            bm_1 = tensor.copy()
            bm_2 = tensor.copy()
            for j in range(resolution):
                for i in [0, 1]:
                    bm_1[j, :, i] = interp1d(tensor[j, :, i])
                    bm_2[:, j, i] = interp1d(tensor[:, j, i])

            with warnings.catch_warnings():
                warnings.filterwarnings(action='ignore', message='Mean of empty slice')
                tensor = np.nanmean(np.stack([bm_1, bm_2]), axis=0)

        results.append(tensor)

    input_map = np.stack(results)
    input_map = rearrange(input_map, 'n h w c -> n c h w')
    input_map = torch.from_numpy(input_map)
    input_map.to(device)
    input_map = input_map.clip(min=0, max=1)

    return input_map



def transform_coords(bm: np.ndarray, coords: np.ndarray) -> np.ndarray:
    dims = check_tensor(bm, "h w 2")
    check_tensor(coords, "n 2")

    height = dims['h']
    width = dims['w']

    points = np.linspace(0, 1, num=height), np.linspace(0, 1, num=width)

    y_channel = scipy.interpolate.interpn(points, bm[..., 0], xi=coords, method='linear')
    x_channel = scipy.interpolate.interpn(points, bm[..., 1], xi=coords, method='linear')

    return np.stack([y_channel, x_channel], axis=-1)


def load_uv_map(uv_file: Path, return_mask: bool = False) -> np.ndarray:
    uv_data = load_npz(uv_file)
    check_tensor(uv_data, "h h c", c=3)

    mask = uv_data[..., 0] > 0.5
    uv_map = uv_data[:, :, 1:]

    uv_map[~mask] = float("NaN")  # mask out background
    uv_map[..., 0] = 1 - uv_map[..., 0]  # invert y-axis (0 should be top)
    return (uv_map, mask) if return_mask else uv_map
