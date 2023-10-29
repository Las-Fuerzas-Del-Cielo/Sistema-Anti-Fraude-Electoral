import cv2

cv2.setNumThreads(0)

import argparse
import os
import sys
from pathlib import Path
from typing import Optional

import gdown
import torch
import tqdm
import yaml
from einops import rearrange

project_dir = Path(__file__).parent.resolve()
sys.path.insert(0, str(project_dir / "src"))

from inv3d_model.models import model_factory
from inv3d_util.image import scale_image
from inv3d_util.load import load_image, save_image, save_npz
from inv3d_util.mapping import apply_map_torch
from inv3d_util.misc import to_numpy_image, to_numpy_map
from inv3d_util.path import list_dirs

model_sources = yaml.safe_load((project_dir / "models.yaml").read_text())


def inference(model_name: str, dataset: str, output_shape: tuple[int, int]):
    model_url = model_sources[model_name]
    model_path = Path(
        gdown.cached_download(
            url=model_url, path=project_dir / f"models/{model_name}.ckpt"
        )
    )

    model = model_factory.load_from_checkpoint(model_name.split("@")[0], model_path)
    model.to("cuda")
    model.eval()

    input_dir = project_dir / "input" / dataset
    output_dir = project_dir / "output" / f"{dataset} - {model_name}"
    output_dir.mkdir(exist_ok=True)

    image_paths = list(input_dir.glob("image_*.*"))

    for image_path in tqdm.tqdm(image_paths, "Unwarping images"):
        sample_name = image_path.stem.removeprefix("image_")

        # prepare image
        image_original = load_image(image_path)
        image = scale_image(
            image_original, resolution=model.dataset_options["resolution"]
        )
        image = rearrange(image, "h w c -> () c h w")
        image = image.astype("float32") / 255
        image = torch.from_numpy(image)
        image = image.to("cuda")

        model_kwargs = {"image": image}

        # prepare template
        if "template" in model_name:
            [template_path] = list(input_dir.glob(f"template_{sample_name}.*"))

            template_original = load_image(template_path)
            template = scale_image(
                template_original, resolution=model.dataset_options["resolution"]
            )
            template = rearrange(template, "h w c -> () c h w")
            template = template.astype("float32") / 255
            template = torch.from_numpy(template)
            template = template.to("cuda")

            model_kwargs["template"] = template

        # inference model
        out_bm = model(**model_kwargs).detach().cpu()

        # unwarp input
        image_original = rearrange(image_original, "h w c -> () c h w")
        image_original = image_original.astype("float32") / 255
        image_original = torch.from_numpy(image_original)

        norm_image = apply_map_torch(
            image=image_original, bm=out_bm, resolution=output_shape
        )

        # export results
        save_image(
            output_dir / f"unwarped_{sample_name}.png",
            to_numpy_image(norm_image),
            override=True,
        )
        save_npz(
            output_dir / f"bm_{sample_name}.npz", to_numpy_map(out_bm), override=True
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        choices=list(model_sources.keys()),
        required=True,
        help="Select the model and the dataset used for training.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=list(map(lambda x: x.name, list_dirs(project_dir / "input"))),
        required=True,
        help="Selects the inference dataset. All folders in the input directory can be selected.",
    )
    parser.add_argument(
        "--output_width",
        type=int,
        default=1700,
        help="Defines the width of the output document in pixels.",
    )
    parser.add_argument(
        "--output_height",
        type=int,
        default=2200,
        help="Defines the height of the output document in pixels.",
    )
    args = parser.parse_args()

    inference(
        model_name=args.model,
        dataset=args.dataset,
        output_shape=(args.output_height, args.output_width),
    )
