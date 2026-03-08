import os
import os.path as osp
import pathlib
import random

import networkx as nx
import numpy as np
import pandas as pd
import torch
import torch_geometric
from hydra.utils import get_original_cwd
from torch_geometric.data import InMemoryDataset
from tqdm import tqdm

from sparse_diffusion.datasets.abstract_dataset import AbstractDataModule, AbstractDatasetInfos
from sparse_diffusion.datasets.dataset_utils import (
    Statistics,
    load_pickle,
    save_pickle,
    RemoveYTransform,
)
from sparse_diffusion.metrics.metrics_utils import node_counts, atom_type_counts, edge_counts
from sparse_diffusion.utils import PlaceHolder


class RedditGraphDataset(InMemoryDataset):
    """SparseDiff-compatible Reddit graph dataset.

    Expected source format is a ``.pt`` file containing tuples ``(A, C, E)`` where:
      - ``A``: node type ids encoded as ``topic + 2``.
      - ``C``: charge ids (ignored by SparseDiff non-molecular setup).
      - ``E``: dense edge matrix with ``1 = NO_BOND``, ``2 = Positive``, ``3 = Negative``.
    """

    def __init__(self, split, root, dataset_file, transform=None, pre_transform=None, pre_filter=None):
        self.split = split
        self.dataset_file = dataset_file
        self.dataset_tag = pathlib.Path(dataset_file).stem
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)

        self.statistics = Statistics(
            num_nodes=load_pickle(self.processed_paths[1]),
            node_types=torch.from_numpy(np.load(self.processed_paths[2])).float(),
            bond_types=torch.from_numpy(np.load(self.processed_paths[3])).float(),
        )

    @property
    def raw_file_names(self):
        return [
            self.dataset_file,
            f"{self.dataset_tag}_train_idx.pt",
            f"{self.dataset_tag}_val_idx.pt",
            f"{self.dataset_tag}_test_idx.pt",
        ]
    
    @property
    def processed_file_names(self):
        return [
            f"{self.dataset_tag}_{self.split}.pt",
            f"{self.dataset_tag}_{self.split}_n.pickle",
            f"{self.dataset_tag}_{self.split}_node_types.npy",
            f"{self.dataset_tag}_{self.split}_bond_types.npy",
        ]

    def download(self):
        # No download URL: user provides dataset_file under self.raw_dir.
        if not osp.exists(self.raw_paths[0]):
            raise FileNotFoundError(
                f"Missing dataset file at {self.raw_paths[0]}. Place your reddit .pt file there."
            )

    def _load_raw_records(self):
        records = torch.load(self.raw_paths[0], weights_only=False)
        if not isinstance(records, list):
            raise ValueError("Reddit dataset file should contain a list of (A, C, E) tuples.")
        return records

    def _build_splits(self):
        records = self._load_raw_records()
        g = torch.Generator()
        g.manual_seed(0)
        indices = torch.randperm(len(records), generator=g)

        test_len = int(round(len(records) * 0.2))
        val_len = int(round((len(records) - test_len) * 0.2))
        train_len = len(records) - val_len - test_len

        train_indices = indices[:train_len]
        val_indices = indices[train_len : train_len + val_len]
        test_indices = indices[train_len + val_len :]

        torch.save(train_indices, self.raw_paths[1])
        torch.save(val_indices, self.raw_paths[2])
        torch.save(test_indices, self.raw_paths[3])

    def process(self):
        if not all(osp.exists(p) for p in self.raw_paths[1:]):
            self._build_splits()

        records = self._load_raw_records()
        split_map = {
            "train": torch.load(self.raw_paths[1]),
            "val": torch.load(self.raw_paths[2]),
            "test": torch.load(self.raw_paths[3]),
        }
        split_indices = split_map[self.split]

        data_list = []
        max_node_type = 0

        for idx in tqdm(split_indices.tolist(), desc=f"Processing {self.split}"):
            A, _C, E = records[idx]
            x = torch.tensor(A, dtype=torch.long) - 2
            max_node_type = max(max_node_type, int(x.max().item()))

            dense_e = torch.tensor(E, dtype=torch.long)
            edge_index = (dense_e > 1).nonzero(as_tuple=False).T.contiguous()

            edge_values = dense_e[edge_index[0], edge_index[1]]
            edge_attr = torch.where(edge_values == 2, 1, 2).long()

            data = torch_geometric.data.Data(
                x=x,
                edge_index=edge_index,
                edge_attr=edge_attr,
                n_nodes=torch.tensor([x.shape[0]], dtype=torch.long),
            )

            if self.pre_filter is not None and not self.pre_filter(data):
                continue
            if self.pre_transform is not None:
                data = self.pre_transform(data)
            data_list.append(data)

        num_node_types = max_node_type + 1
        num_nodes = node_counts(data_list)
        node_types = atom_type_counts(data_list, num_classes=num_node_types)
        bond_types = edge_counts(data_list, num_bond_types=3)

        torch.save(self.collate(data_list), self.processed_paths[0])
        save_pickle(num_nodes, self.processed_paths[1])
        np.save(self.processed_paths[2], node_types)
        np.save(self.processed_paths[3], bond_types)


class RedditDatasetBuilder:
    """Utility to build reddit ``(A, C, E)`` records from the original CSV file."""

    def __init__(self):
        self.charges = {-999: 0, -998: 1, 0: 2}
        self.bonds = {"MASK": 0, "NO_BOND": 1, "Positive": 2, "Negative": 3}

    def preprocess(self, csv_path):
        df = pd.read_csv(csv_path)
        df = df[(df["source_type"] != -1) & (df["target_type"] != -1)]
        df = df.drop(
            columns=["POST_ID", "TIMESTAMP", "PROPERTIES", "source_topic", "target_topic"]
        )
        output_path = osp.join(osp.dirname(csv_path), "giant_preprocess.csv")
        df.to_csv(output_path, index=False)
        return output_path

    def build_graph(self, csv_path):
        df = pd.read_csv(csv_path)
        g = nx.from_pandas_edgelist(
            df,
            "SOURCE_SUBREDDIT",
            "TARGET_SUBREDDIT",
            edge_attr=["LINK_SENTIMENT"],
            create_using=nx.Graph(),
        )

        for _, row in df.iterrows():
            g.nodes[row["SOURCE_SUBREDDIT"]]["topic"] = row["source_type"]
            g.nodes[row["TARGET_SUBREDDIT"]]["topic"] = row["target_type"]

        return g

    def sample_subgraph(self, graph, p_restart=0.3, max_size=800, seed=None):
        if seed is not None:
            random.seed(seed)

        topics = list(set(nx.get_node_attributes(graph, "topic").values()))
        chosen_topic = random.choice(topics)
        candidates = [n for n, d in graph.nodes(data=True) if d.get("topic") == chosen_topic]
        start = random.choice(candidates)

        current = start
        visited = {start}

        while len(visited) < max_size:
            if random.random() < p_restart:
                current = start
            else:
                neighbors = list(graph.neighbors(current))
                if neighbors:
                    current = random.choice(neighbors)
            visited.add(current)

        return graph.subgraph(visited).copy()

    def nx_to_ace(self, subgraph):
        nodes = list(subgraph.nodes())
        n = len(nodes)
        node_idx = {node: i for i, node in enumerate(nodes)}

        a = [subgraph.nodes[node]["topic"] + 2 for node in nodes]
        c = [self.charges[0]] * n

        e = np.full((n, n), self.bonds["NO_BOND"], dtype=np.int64)
        for u, v, data in subgraph.edges(data=True):
            sentiment = data["LINK_SENTIMENT"]
            bond_type = self.bonds["Positive"] if sentiment > 0 else self.bonds["Negative"]
            e[node_idx[u]][node_idx[v]] = bond_type
            e[node_idx[v]][node_idx[u]] = bond_type

        return a, c, e

    def build_dataset_file(
        self,
        csv_path,
        output_path,
        num_subgraphs=500,
        subgraph_size=200,
        base_seed=42,
    ):
        preprocessed_path = self.preprocess(csv_path)
        gcc = self.build_graph(preprocessed_path)

        records = []
        for i in tqdm(range(num_subgraphs), desc="Sampling reddit subgraphs"):
            subgraph = self.sample_subgraph(gcc, max_size=subgraph_size, seed=i + base_seed)
            records.append(self.nx_to_ace(subgraph))

        os.makedirs(osp.dirname(output_path), exist_ok=True)
        torch.save(records, output_path)
        return output_path


class RedditDataModule(AbstractDataModule):
    def __init__(self, cfg):
        self.cfg = cfg
        self.dataset_name = self.cfg.dataset.name
        self.datadir = cfg.dataset.datadir
        self.dataset_file = cfg.dataset.dataset_file

        base_path = pathlib.Path(get_original_cwd()).parents[0]
        root_path = os.path.join(base_path, self.datadir)

        pre_transform = RemoveYTransform()

        datasets = {
            "train": RedditGraphDataset(
                split="train",
                root=root_path,
                dataset_file=self.dataset_file,
                pre_transform=pre_transform,
            ),
            "val": RedditGraphDataset(
                split="val",
                root=root_path,
                dataset_file=self.dataset_file,
                pre_transform=pre_transform,
            ),
            "test": RedditGraphDataset(
                split="test",
                root=root_path,
                dataset_file=self.dataset_file,
                pre_transform=pre_transform,
            ),
        }

        self.statistics = {
            "train": datasets["train"].statistics,
            "val": datasets["val"].statistics,
            "test": datasets["test"].statistics,
        }

        super().__init__(cfg, datasets)
        super().prepare_dataloader()
        self.inner = self.train_dataset


class RedditDatasetInfos(AbstractDatasetInfos):
    def __init__(self, datamodule):
        self.is_molecular = False
        self.spectre = True
        self.use_charge = False
        self.dataset_name = datamodule.dataset_name

        self.node_types = datamodule.inner.statistics.node_types
        self.bond_types = datamodule.inner.statistics.bond_types
        super().complete_infos(datamodule.statistics, len(self.node_types))

        self.input_dims = PlaceHolder(X=len(self.node_types), E=len(self.bond_types), y=0, charge=0)
        self.output_dims = PlaceHolder(X=len(self.node_types), E=len(self.bond_types), y=0, charge=0)
        self.statistics = {
            "train": datamodule.statistics["train"],
            "val": datamodule.statistics["val"],
            "test": datamodule.statistics["test"],
        }