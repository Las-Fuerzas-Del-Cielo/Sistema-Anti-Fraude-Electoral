import os

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

import json
from pathlib import Path
from typing import *

import cv2
import numpy as np
import yaml
import h5py

from .path import check_file


def load_image(file: Path):
    file = check_file(file, suffix=[".png", ".jpg"])
    return cv2.imread(str(file), cv2.IMREAD_COLOR)


def save_image(file: Path, data: np.ndarray, override: bool = False):
    exist = None if override else False
    file = check_file(file, suffix=[".png", ".jpg"], exist=exist)
    assert data.dtype == np.uint8
    assert len(data.shape) in [2, 3]
    if len(data.shape) == 3:
        assert data.shape[2] == 3

    cv2.imwrite(str(file), data)


def load_npz(file: Path) -> np.ndarray:
    file = check_file(file, suffix=".npz")
    with np.load(file) as archive:
        keys = list(archive.keys())
        assert len(keys) == 1
        return archive[keys[0]]


def load_exr(file: Path):
    file = check_file(file, suffix=".exr")
    return cv2.imread(str(file.resolve()), cv2.IMREAD_UNCHANGED)


def load_mat(file: Path, key: str = None):
    file = check_file(file, suffix=".mat")

    with h5py.File(str(file.resolve()), 'r') as f:
        if key is None:
            key = list(f.keys())[0]
        return np.array(f[key])


def save_npz(file: Path, data: np.ndarray, override: bool = False):
    exist = None if override else False
    file = check_file(file, suffix=".npz", exist=exist)

    params = {
        file.stem: data
    }
    np.savez_compressed(str(file), **params)


def load_array(file: Path):
    file = Path(file)

    if file.suffix == ".npz":
        return load_npz(file)

    if file.suffix == ".exr":
        return load_exr(file)

    if file.suffix == ".mat":
        return load_mat(file)

    assert False, f"Cannot load array! Unknown file extension {file.suffix}!"


def load_json(file: Union[str, Path]):
    file = check_file(file, suffix=".json", exist=True)
    with file.open("r") as fp:
        return json.load(fp)


def save_json(file: Path, data: Dict, exist=False):
    file = check_file(file, suffix=".json", exist=exist)
    with file.open("w") as f:
        json.dump(obj=data, fp=f, indent=4)


def load_yaml(file: Union[str, Path]):
    file = check_file(file, suffix=".yaml", exist=True)
    with file.open("r") as fp:
        return yaml.safe_load(fp)


def save_yaml(file: Path, data: Dict, exist=False):
    file = check_file(file, suffix=".yaml", exist=exist)
    with file.open("w") as f:
        yaml.dump(data, f, default_flow_style=False)
