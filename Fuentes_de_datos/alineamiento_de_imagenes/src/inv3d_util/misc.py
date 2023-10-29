from .load import load_image, load_npz
import re
from pathlib import Path
from typing import *

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange

plt.rcParams['figure.figsize'] = [10, 10]


def check_tensor(data, pattern: str, allow_none: bool = False, **kwargs):
    if allow_none and data is None:
        return {}

    assert bool(re.match('^[a-zA-Z0-9 ]+$', pattern)), "Invalid characters in pattern found! Only use [a-zA-Z0-9 ]."

    tokens = [t for t in pattern.split(" ") if t != '']

    assert len(data.shape) == len(tokens), "Input tensor has an invalid number of dimensions!"

    assignment = {}
    for dim, (token, size) in enumerate(zip(tokens, data.shape)):
        if token[0].isdigit():
            assert int(token) == size, f"Tensor mismatch in dimension {dim}: expected {size}, found {int(token)}!"
        else:
            if token in assignment:
                assert assignment[
                    token] == size, f"Tensor mismatch in dimension {dim}: expected {size}, found {assignment[token]}!"
            else:
                assignment[token] = size

                if token in kwargs:
                    assert kwargs[
                        token] == size, f"Tensor mismatch in dimension {dim}: expected {kwargs[token]}, found {size}!"

    return assignment


def to_numpy_image(image: Union[Path, np.ndarray, torch.Tensor]):
    if isinstance(image, Path):
        image = load_image(image)
    elif isinstance(image, torch.Tensor):
        image = image.detach().cpu().numpy()

    if np.issubdtype(image.dtype, np.floating):
        image = (image * 255).astype("uint8")

    image = np.squeeze(image)

    if image.shape[0] <= 3:
        image = rearrange(image, "c h w -> h w c")

    dims = check_tensor(image, "h w c")
    assert dims["c"] <= 3

    padding = np.zeros((dims["h"], dims["w"], 3 - dims["c"]), dtype=np.uint8)
    return np.concatenate([image, padding], axis=-1)


def to_numpy_map(input_map: Union[Path, np.ndarray, torch.Tensor]):
    if isinstance(input_map, Path):
        input_map = load_npz(input_map)
    elif isinstance(input_map, torch.Tensor):
        input_map = input_map.detach().cpu().numpy()

    input_map = np.squeeze(input_map)

    if input_map.shape[0] == 2:
        input_map = rearrange(input_map, "c h w -> h w c")

    check_tensor(input_map, "h w 2")
    return input_map


def median_blur(x: torch.Tensor):
    dims = check_tensor(x, "n c h w")
    assert dims["c"] in [2, 3]

    kernel = torch.ones((1, 1, 3, 3)) / 9
    if x.is_cuda:
        kernel = kernel.to(x.get_device())

    x = rearrange(x, "n c h w -> (n c) 1 h w")
    x = torch.nn.ReplicationPad2d(1)(x)
    x = F.conv2d(x, kernel)
    x = rearrange(x, "(n c) 1 h w -> n c h w", c=dims["c"])

    return x