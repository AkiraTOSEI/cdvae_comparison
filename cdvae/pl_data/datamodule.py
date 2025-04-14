import random
from typing import Optional, Sequence
from pathlib import Path

import hydra
import numpy as np
import omegaconf
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig
from torch.utils.data import Dataset
from torch_geometric.data import DataLoader
from torch.utils.data import ConcatDataset
from cdvae.common.data_utils import get_scaler_from_data_list, IdentityScaler

from cdvae.common.utils import PROJECT_ROOT
from cdvae.common.data_utils import get_scaler_from_data_list


def worker_init_fn(id: int):
    """
    DataLoaders workers init function.

    Initialize the numpy.random seed correctly for each worker, so that
    random augmentations between workers and/or epochs are not identical.

    If a global seed is set, the augmentations are deterministic.

    https://pytorch.org/docs/stable/notes/randomness.html#dataloader
    """
    uint64_seed = torch.initial_seed()
    ss = np.random.SeedSequence([uint64_seed])
    # More than 128 bits (4 32-bit words) would be overkill.
    np.random.seed(ss.generate_state(4))
    random.seed(uint64_seed)

def collate_skip_none(batch):
    return DataLoader.collate([b for b in batch if b is not None])

class CrystDataModule(pl.LightningDataModule):
    def __init__(
        self,
        datasets: DictConfig,
        num_workers: DictConfig,
        batch_size: DictConfig,
        scaler_path=None,
    ):
        super().__init__()
        self.datasets = datasets
        self.num_workers = num_workers
        self.batch_size = batch_size

        self.train_dataset: Optional[Dataset] = None
        self.val_datasets: Optional[Sequence[Dataset]] = None
        self.test_datasets: Optional[Sequence[Dataset]] = None

        self.get_scaler(scaler_path)

    def prepare_data(self) -> None:
        # download only
        pass

    def get_scaler(self, scaler_path):
        # scaler_path が None の場合、train dataset からスケーラーを作成する
        if scaler_path is None:
            # _recursive_=True を使うと、設定値が再帰的に反映されます
            train_dataset = hydra.utils.instantiate(self.datasets.train, _recursive_=True)
            self.lattice_scaler = get_scaler_from_data_list(
                train_dataset.cached_data,
                key='scaled_lattice'
            )
            print(f"[DEBUG] train_dataset.task: {train_dataset.task}", flush=True)
            
            # task の値でスケーラーの生成方法を分岐
            if train_dataset.task in ["megnet", "megnet_perov"]:
                print(f"[DEBUG] Using special scaler; dataset.prop: {train_dataset.prop}", flush=True)
                scaler = []
                for p in train_dataset.prop:
                    if p in ['100more', 'tolerance']:
                        # これらのプロパティは IdentityScaler を使用する
                        scaler.append(IdentityScaler())
                    else:
                        scaler.append(get_scaler_from_data_list(train_dataset.cached_data, key=p))
                self.scaler = scaler
            else:
                # 通常の場合
                if isinstance(train_dataset.prop, list):
                    self.scaler = [
                        get_scaler_from_data_list(train_dataset.cached_data, key=p)
                        for p in train_dataset.prop
                    ]
                else:
                    self.scaler = get_scaler_from_data_list(train_dataset.cached_data, key=train_dataset.prop)
        else:
            # スケーラーが保存されたパスが指定されている場合はそれをロード
            self.lattice_scaler = torch.load(Path(scaler_path) / 'lattice_scaler.pt')
            self.scaler = torch.load(Path(scaler_path) / 'prop_scaler.pt')


    def setup(self, stage: Optional[str] = None):
        """
        construct datasets and assign data scalers.
        """
        if stage is None or stage == "fit":
            self.train_dataset = hydra.utils.instantiate(self.datasets.train)
            self.val_datasets = [
                hydra.utils.instantiate(dataset_cfg)
                for dataset_cfg in self.datasets.val
            ]
            print("✅ self.val_datasets:", self.val_datasets)


            self.train_dataset.lattice_scaler = self.lattice_scaler
            self.train_dataset.scaler = self.scaler
            for val_dataset in self.val_datasets:
                val_dataset.lattice_scaler = self.lattice_scaler
                val_dataset.scaler = self.scaler

        if stage is None or stage == "test":
            self.test_datasets = [
                hydra.utils.instantiate(dataset_cfg)
                for dataset_cfg in self.datasets.test
            ]
            for test_dataset in self.test_datasets:
                test_dataset.lattice_scaler = self.lattice_scaler
                test_dataset.scaler = self.scaler

                

    def train_dataloader(self) -> DataLoader:
        valid_data = [d for d in (self.train_dataset[i] for i in range(len(self.train_dataset))) if d is not None]
        return DataLoader(
            valid_data,
            shuffle=True,
            batch_size=self.batch_size.train,
            num_workers=self.num_workers.train,
            worker_init_fn=worker_init_fn,
            collate_fn=collate_skip_none,  # ← これを追加
        )

    def val_dataloader(self) -> Sequence[DataLoader]:
        """
        return [
            DataLoader(
                dataset,
                shuffle=False,
                batch_size=self.batch_size.val,
                num_workers=self.num_workers.val,
                worker_init_fn=worker_init_fn,
            )
            for dataset in self.val_datasets
        ]"""
        #concat_val_dataset = ConcatDataset(self.val_datasets)
        valid_data = []
        for dataset in self.val_datasets:
            for i in range(len(dataset)):
                d = dataset[i]
                if d is not None:
                    valid_data.append(d)
        return DataLoader(
            valid_data,
            shuffle=False,
            batch_size=self.batch_size.val,
            num_workers=self.num_workers.val,
            worker_init_fn=worker_init_fn,
            collate_fn=collate_skip_none,  # ← これを追加
        )

    def test_dataloader(self) -> Sequence[DataLoader]:
        loaders = []
        for dataset in self.test_datasets:
            valid_data = [d for d in (dataset[i] for i in range(len(dataset))) if d is not None]
            loader = DataLoader(
                valid_data,
                shuffle=False,
                batch_size=self.batch_size.test,
                num_workers=self.num_workers.test,
                worker_init_fn=worker_init_fn,
                collate_fn=collate_skip_none,
            )
            loaders.append(loader)
        return loaders

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"{self.datasets=}, "
            f"{self.num_workers=}, "
            f"{self.batch_size=})"
        )


@hydra.main(config_path=str(PROJECT_ROOT / "conf"), config_name="default")
def main(cfg: omegaconf.DictConfig):
    datamodule: pl.LightningDataModule = hydra.utils.instantiate(
        cfg.data.datamodule, _recursive_=False
    )
    datamodule.setup('fit')
    import pdb
    pdb.set_trace()


if __name__ == "__main__":
    main()
