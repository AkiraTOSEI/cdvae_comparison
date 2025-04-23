import hydra
import omegaconf
import torch
import pandas as pd
from omegaconf import ValueNode
from torch.utils.data import Dataset

from torch_geometric.data import Data

from cdvae.common.utils import PROJECT_ROOT
from cdvae.common.data_utils import (
    preprocess, preprocess_tensors, add_scaled_lattice_prop)


class CrystDataset(Dataset):
    def __init__(self, name: ValueNode, path: ValueNode,
                 prop: ValueNode, niggli: ValueNode, primitive: ValueNode,
                 graph_method: ValueNode, preprocess_workers: ValueNode,
                 lattice_scale_method: ValueNode,
                 task: ValueNode,
                 **kwargs):
        super().__init__()
        self.path = path
        self.name = name
        self.df = pd.read_csv(path)
        self.prop = prop
        self.niggli = niggli
        self.primitive = primitive
        self.graph_method = graph_method
        self.lattice_scale_method = lattice_scale_method

        self.task = task
        if self.task == "megnet":
            self.prop = ["gap", "e_form", "100more", "tolerance"]
        else:
            self.prop = prop

        prop_list = self.prop if isinstance(self.prop, list) else [self.prop]

        self.cached_data = preprocess(
            self.path,
            preprocess_workers,
            niggli=self.niggli,
            primitive=self.primitive,
            graph_method=self.graph_method,
            #prop_list=[self.prop])
            prop_list=  prop_list)

        add_scaled_lattice_prop(self.cached_data, lattice_scale_method)
        self.lattice_scaler = None
        self.scaler = None

    def __len__(self) -> int:
        return len(self.cached_data)

    def __getitem__(self, index):
        data_dict = self.cached_data[index]
        print(f"[DEBUG] data_dict: {data_dict.keys()}")
        # scaler is set in DataModule set stage
        if self.task == "megnet":
            # 各カラムに対応するスケーラーで変換
            prop = torch.tensor([
                self.scaler[i].transform(data_dict[p])
                for i, p in enumerate(self.prop)
            ], dtype=torch.float)
        else:
            prop = self.scaler.transform(data_dict[self.prop])
        (frac_coords, atom_types, lengths, angles, edge_indices,
         to_jimages, num_atoms) = data_dict['graph_arrays']

        # atom_coords are fractional coordinates
        # edge_index is incremented during batching
        # https://pytorch-geometric.readthedocs.io/en/latest/notes/batching.html
        print(f"\n🧪 index {index}")
        print(f"  - num_atoms: {num_atoms}")
        print(f"  - atom_types: {atom_types}")
        print(f"  - frac_coords shape: {frac_coords.shape}")
        print(f"  - edge_indices shape: {edge_indices.shape}")
        print(f"  - to_jimages shape: {to_jimages.shape}")
        print(f"  - lengths: {lengths}")
        print(f"  - angles: {angles}")
        try:
            data = Data(
                frac_coords=torch.Tensor(frac_coords),
                atom_types=torch.LongTensor(atom_types),
                lengths=torch.Tensor(lengths).view(1, -1),
                angles=torch.Tensor(angles).view(1, -1),
                edge_index=torch.LongTensor(
                    edge_indices.T).contiguous(),  # shape (2, num_edges)
                to_jimages=torch.LongTensor(to_jimages),
                num_atoms=num_atoms,
                num_bonds=edge_indices.shape[0],
                num_nodes=num_atoms,  # special attribute used for batching in pytorch geometric
                y=prop.view(1, -1),
            )
            return data
        except Exception as e:
            print(f"🛑 Skipping index {index} due to exception in __getitem__: {e}")
            return None  # PyGのBatchではNoneがあるとエラーになるので注意（次で除去する）

    def __repr__(self) -> str:
        return f"CrystDataset({self.name=}, {self.path=})"


class TensorCrystDataset(Dataset):
    def __init__(self, crystal_array_list, niggli, primitive,
                 graph_method, preprocess_workers,
                 lattice_scale_method, **kwargs):
        super().__init__()
        self.niggli = niggli
        self.primitive = primitive
        self.graph_method = graph_method
        self.lattice_scale_method = lattice_scale_method

        self.cached_data = preprocess_tensors(
            crystal_array_list,
            niggli=self.niggli,
            primitive=self.primitive,
            graph_method=self.graph_method,
            num_cpus=preprocess_workers
            )

        add_scaled_lattice_prop(self.cached_data, lattice_scale_method)
        self.lattice_scaler = None
        self.scaler = None

    def __len__(self) -> int:
        return len(self.cached_data)

    def __getitem__(self, index):
        data_dict = self.cached_data[index]

        (frac_coords, atom_types, lengths, angles, edge_indices,
         to_jimages, num_atoms) = data_dict['graph_arrays']

        # atom_coords are fractional coordinates
        # edge_index is incremented during batching
        # https://pytorch-geometric.readthedocs.io/en/latest/notes/batching.html
        data = Data(
            frac_coords=torch.Tensor(frac_coords),
            atom_types=torch.LongTensor(atom_types),
            lengths=torch.Tensor(lengths).view(1, -1),
            angles=torch.Tensor(angles).view(1, -1),
            edge_index=torch.LongTensor(
                edge_indices.T).contiguous(),  # shape (2, num_edges)
            to_jimages=torch.LongTensor(to_jimages),
            num_atoms=num_atoms,
            num_bonds=edge_indices.shape[0],
            num_nodes=num_atoms,  # special attribute used for batching in pytorch geometric
        )
        return data

    def __repr__(self) -> str:
        return f"TensorCrystDataset(len: {len(self.cached_data)})"


@hydra.main(config_path=str(PROJECT_ROOT / "conf"), config_name="default")
def main(cfg: omegaconf.DictConfig):
    from torch_geometric.data import Batch
    from cdvae.common.data_utils import get_scaler_from_data_list, IdentityScaler
    dataset: CrystDataset = hydra.utils.instantiate(
        cfg.data.datamodule.datasets.train, _recursive_=False
    )
    lattice_scaler = get_scaler_from_data_list(
        dataset.cached_data,
        key='scaled_lattice')
    print(f"[DEBUG] dataset.task: {dataset.task}")
    if dataset.task == "megnet" or dataset.task == 'megnet_perov':
        print(f"dataset.prop: {dataset.prop}")
        print('use special scaler')
        scaler = []
        for p in dataset.prop:
            if p in ['100more', 'tolerance']:
                scaler.append(IdentityScaler())  # ← この2つにはスケーラーを適用しない
            else:
                scaler.append(get_scaler_from_data_list(dataset.cached_data, key=p))
    else:
        scaler = get_scaler_from_data_list(
            dataset.cached_data,
            key=dataset.prop)

    dataset.lattice_scaler = lattice_scaler
    dataset.scaler = scaler
    data_list = [dataset[i] for i in range(len(dataset))]
    batch = Batch.from_data_list(data_list)
    return batch


if __name__ == "__main__":
    main()
