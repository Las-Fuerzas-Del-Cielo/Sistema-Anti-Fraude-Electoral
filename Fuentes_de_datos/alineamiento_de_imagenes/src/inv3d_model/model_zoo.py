import cv2

cv2.setNumThreads(0)

import shutil
import pandas as pd
import tqdm
import re
import os
import inspect
import resource

from copy import deepcopy
from dataclasses import dataclass
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.utilities.seed import seed_everything
from torch.utils.data import DataLoader

from inv3d_util.load import save_image, save_npz
from inv3d_util.misc import to_numpy_image, to_numpy_map
from inv3d_util.mapping import apply_map_torch
from inv3d_util.path import *
from inv3d_util.visualization import visualize_image, visualize_bm

from .score.score import *
from .models import model_factory
from .datasets.dataset_factory import DatasetFactory, DatasetSplit


seed_everything(seed=42, workers=True)

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"


class ModelZoo:

    def __init__(self, root_dir: Path, sources_file: Path):
        self.root_dir = Path(root_dir)
        self.dataset_factory = DatasetFactory(sources_file)

        # torch.multiprocessing.set_sharing_strategy('file_system')
        resource.setrlimit(resource.RLIMIT_NOFILE, (65535, 65535))

    def list_models(self):
        for model in model_factory.get_all_models():
            print(model)

    def list_datasets(self):
        for dataset in self.dataset_factory.get_all_datasets():
            print(dataset)

    def list_trained_models(self):
        for d in list_dirs(self.root_dir):
            print(d.stem)

    def load_model(self, name: str) -> LightningModule:
        run_config = RunConfig.from_str(name)

        checkpoints = (self.root_dir / name).rglob("checkpoint-epoch=*.ckpt")
        checkpoint = max(checkpoints, key=lambda x: int(x.stem.replace("=", "-").split("-")[2]))

        print(f"Loading checkpoint: {str(checkpoint.resolve())}")

        return model_factory.load_from_checkpoint(run_config.model, checkpoint)

    def delete_model(self, name: str):
        model_dir = self.root_dir / name
        if model_dir.is_dir():
            shutil.rmtree(str(model_dir))

    def train_model(self, name: str, gpus: Union[int, Iterable[int]], num_workers: int, fast_dev_run: bool = False,
                    model_kwargs: Optional[Dict] = None, resume: bool = False):

        run_config = RunConfig.from_str(name)

        output_dir = check_dir(self.root_dir / name, exist=resume)
        output_dir.mkdir(exist_ok=True)

        # create model
        model_kwargs = {} if model_kwargs is None else model_kwargs
        model = model_factory.create_new(run_config.model, **model_kwargs)

        if not resume:
            shutil.copyfile(inspect.getsourcefile(model.__class__), output_dir / "model.py")

        # gather options
        train_options = deepcopy(model.train_options)
        batch_size = train_options.pop("batch_size")
        max_epochs = train_options.pop("max_epochs")
        patience = train_options.pop("early_stopping_patience", None)
        gpus = [gpus] if isinstance(gpus, int) else list(gpus)

        # prepare gpu configuration
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpus))

        # create datasets
        train_dataset = self.dataset_factory.create(name=run_config.dataset,
                                                    split=DatasetSplit.TRAIN,
                                                    limit_samples=run_config.limit_samples,
                                                    repeat_samples=run_config.repeat_samples,
                                                    **model.dataset_options)

        val_dataset = self.dataset_factory.create(name=run_config.dataset,
                                                  split=DatasetSplit.VALIDATE,
                                                  limit_samples=run_config.limit_samples,
                                                  **model.dataset_options)

        train_loader = DataLoader(dataset=train_dataset,
                                  batch_size=batch_size,
                                  num_workers=num_workers,
                                  shuffle=True,
                                  drop_last=True,
                                  persistent_workers=True)

        val_loader = DataLoader(dataset=val_dataset,
                                batch_size=batch_size,
                                num_workers=num_workers,
                                shuffle=False,
                                drop_last=True,
                                persistent_workers=True)

        # update properties required for training scheduler
        model.epochs = max_epochs
        model.steps_per_epoch = len(train_loader) // len(gpus)

        # create callbacks
        callbacks = [ModelCheckpoint(dirpath=output_dir / "checkpoints",
                                     monitor='val/mse_loss', save_top_k=1, mode='min',
                                     filename='checkpoint-epoch={epoch:002d}-val_mse_loss={val/mse_loss:.4f}',
                                     auto_insert_metric_name=False,
                                     save_last=True),
                     LearningRateMonitor()]

        if patience:
            callbacks.append(EarlyStopping(monitor='val/mse_loss', patience=patience, mode='min', verbose=True))

        logger = TensorBoardLogger(output_dir, name="logs", version="")

        # create trainer
        trainer = Trainer(gpus=-1,
                          logger=logger,
                          fast_dev_run=fast_dev_run,
                          callbacks=callbacks,
                          max_epochs=max_epochs,
                          **train_options)

        resume_kwargs = {"ckpt_path": output_dir / "checkpoints/last.ckpt"} if resume else {}

        trainer.fit(model, train_loader, val_loader, **resume_kwargs)

    def eval_model(self, name: str, dataset_name: str, gpu: int, num_workers: int = 4):
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)

        eval_dir = check_dir(self.root_dir / name) / "eval" / dataset_name
        eval_dir.mkdir(parents=True)

        model = self.load_model(name)
        model.cuda()
        model.eval()

        eval_dataset = self.dataset_factory.create(name=dataset_name,
                                                   split=DatasetSplit.TEST,
                                                   **model.dataset_options)

        eval_loader = DataLoader(eval_dataset,
                                 batch_size=1,  # batch size 1 is required due to changing aspect ratios!
                                 num_workers=num_workers,
                                 shuffle=False,
                                 drop_last=False,
                                 persistent_workers=True)

        states = []
        with tqdm.tqdm(total=len(eval_loader), desc="Inference samples") as progress:
            for index, batch in enumerate(eval_loader):
                input_data = {key: value.cuda() for key, value in batch["input"].items()}

                out_bm = model(**input_data).detach().cpu()
                true_bm = batch["eval"].get("true_bm", None)

                orig_image = batch["eval"]["orig_image"]
                true_image = batch["eval"]["true_image"]
                norm_image = apply_map_torch(image=orig_image, bm=out_bm, resolution=true_image.shape[-2:])

                index = int(batch["index"][0])

                states.append({
                    "out_bm": out_bm,
                    "true_bm": true_bm,
                    "norm_image": norm_image,
                    "true_image": true_image,
                    "orig_image": orig_image,
                    "text_evaluation": batch["eval"]["text_evaluation"][0],
                    "results": {
                        "eval_dataset": dataset_name,
                        "sample": batch["sample"][0],
                        "index": index
                    }
                })

                if index < 100:
                    output_dir = eval_dir / "examples" / str(index)
                    output_dir.mkdir(parents=True)
                    save_image(output_dir / "orig_image.png", to_numpy_image(orig_image))
                    save_image(output_dir / "true_image.png", to_numpy_image(true_image))
                    save_image(output_dir / "norm_image.png", to_numpy_image(norm_image))

                    save_npz(output_dir / "out_bm.npz", to_numpy_map(out_bm))
                    if true_bm is not None:
                        save_npz(output_dir / "true_bm.npz", to_numpy_map(true_bm))

                progress.update(1)

        df = score_all(states=states)

        df.to_csv(eval_dir / "results.csv", index=False)

        self.show_results(name, dataset_name)

    def inference(self, name: str, dataset_name: str, gpu: int, num_workers: int = 4):
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)

        inference_dir = check_dir(self.root_dir / name) / "inference" / dataset_name
        inference_dir.mkdir(parents=True)

        model = self.load_model(name)
        model.cuda()
        model.eval()

        inference_dataset = self.dataset_factory.create(name=dataset_name,
                                                        split=DatasetSplit.TEST,
                                                        **model.dataset_options)

        inference_loader = DataLoader(inference_dataset,
                                      batch_size=1,  # batch size 1 is required due to changing aspect ratios!
                                      num_workers=num_workers,
                                      shuffle=False,
                                      drop_last=False,
                                      persistent_workers=True)

        with tqdm.tqdm(total=len(inference_loader), desc="Inference samples") as progress:
            for batch in inference_loader:
                input_data = {key: value.cuda() for key, value in batch["input"].items()}

                out_bm = model(**input_data).detach().cpu()
                    
                orig_image = batch["eval"]["orig_image"]
                true_image = batch["eval"]["true_image"]
                norm_image = apply_map_torch(image=orig_image, bm=out_bm, resolution=true_image.shape[-2:])

                [sample] = batch["sample"]
                sample = Path(sample)

                output_dir = inference_dir / sample.parent.name / sample.stem
                output_dir.mkdir(parents=True)
                save_image(output_dir / "orig_image.png", to_numpy_image(orig_image))
                save_image(output_dir / "true_image.png", to_numpy_image(true_image))
                save_image(output_dir / "norm_image.png", to_numpy_image(norm_image))
                # shutil.copyfile(str(sample.parent / "flat_template.png"), str(output_dir / "template.png")) # TODO remove
                save_npz(output_dir / "out_bm.npz", to_numpy_map(out_bm))

                progress.update(1)


    def show_sample(self, name: str, dataset_name: str, idx: int):
        example_dir = self.root_dir / name / "eval" / dataset_name / "examples" / str(idx)

        visualize_image(example_dir / "orig_image.png")
        visualize_image(example_dir / "true_image.png")
        visualize_image(example_dir / "norm_image.png")

        if (example_dir / "true_bm.npz").is_file():
            visualize_bm(example_dir / "orig_image.png", example_dir / "true_bm.npz", title="True backward map")

        visualize_bm(example_dir / "orig_image.png", example_dir / "out_bm.npz", title="Output backward map")

    def show_results(self, name: str, dataset_name: str):
        eval_dir = self.root_dir / name / "eval" / dataset_name

        df = pd.read_csv(eval_dir / "results.csv")

        print(f"Results for model '{name}' on '{dataset_name}':")
        print(df.drop(columns=["sample", "index"]).mean())

        return df

    def show_all_results(self, evaluation: str = None, group: bool = True):
        dfs = []
        for file in self.root_dir.rglob("results.csv"):
            df = pd.read_csv(file)
            df["model"] = file.parts[-4]
            df["evaluation"] = file.parts[-2]
            dfs.append(df)

        if len(dfs) == 0:
            return pd.DataFrame()

        df = pd.concat(dfs)
        df = df.drop(columns=["index"])

        if evaluation is not None:
            df = df[df.evaluation == evaluation]

        if group:
            df = df.groupby(["model", "evaluation"]).agg(['mean', 'std'])

        return df


@dataclass
class RunConfig:
    model: str
    dataset: Optional[str]
    limit_samples: Optional[str]
    repeat_samples: Optional[str]
    version: Optional[str]

    @staticmethod
    def from_str(name: str) -> "RunConfig":
        pattern = r"^(?P<model>[^\s@]+)(@(?P<dataset>[^\s@\[\]]+)(\[(?P<limit_samples>\d+)(x(?P<repeat_samples>\d+))?\])?(@(?P<version>[^\s@]+))?)?$"
        match = re.match(pattern, name)

        if match is None:
            raise ValueError(f"Invalid model name! Could not parse '{name}'!")

        limit_samples = match.groupdict().get("limit_samples", None)
        limit_samples = None if limit_samples is None else int(limit_samples)

        repeat_samples = match.groupdict().get("repeat_samples", None)
        repeat_samples = None if repeat_samples is None else int(repeat_samples)

        return RunConfig(
            model=match.groupdict()["model"],
            dataset=match.groupdict().get("dataset", None),
            limit_samples=limit_samples,
            repeat_samples=repeat_samples,
            version=match.groupdict().get("version", None)
        )
