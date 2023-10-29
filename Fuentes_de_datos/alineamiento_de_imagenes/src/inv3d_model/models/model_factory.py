from pathlib import Path
from typing import Type, List
from pytorch_lightning import LightningModule

_all_models: dict[str, type[LightningModule]] = {}


def get_all_models() -> List[str]:
    return sorted(list(_all_models.keys()))


def register_model(name: str, model_class: type[LightningModule]):
    _all_models[name] = model_class


def class_by_name(name: str) -> type[LightningModule]:
    if name not in _all_models:
        raise ValueError(f"Model '{name}' is unknown! Cannot find class!")

    return _all_models[name]


def create_new(name: str, **model_kwargs) -> LightningModule:
    if name not in _all_models:
        raise ValueError(f"Model '{name}' is unknown! Cannot create a new instance!")

    return _all_models[name](**model_kwargs)


def load_from_checkpoint(name: str, checkpoint_file: Path) -> LightningModule:
    if name not in _all_models:
        raise ValueError(
            f"Model '{name}' is unknown! Cannot load the model from checkpoint!"
        )

    if not checkpoint_file.is_file():
        raise ValueError(
            f"Model checkpoint '{str(checkpoint_file.resolve())}' does not exist! Cannot load the model from checkpoint!"
        )

    return _all_models[name].load_from_checkpoint(str(checkpoint_file))
