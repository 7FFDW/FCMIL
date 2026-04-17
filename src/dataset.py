import torch
from torch.utils.data import Dataset
from typing import List, Dict, Any, Tuple, Union
from dataclasses import dataclass
import logging


@dataclass
class DatasetConfig:
    """Configuration for dataset initialization."""
    data_root: str
    label_list: List[str]
    mapping: str
    
    def __post_init__(self):
        """Parse mapping string into dictionary after initialization."""
        try:
            pairs = self.mapping.split(", ")
            self.mapping = {int(k): int(v) for k, v in (pair.split(":") for pair in pairs)}
        except (ValueError, KeyError) as e:
            raise ValueError(f"Invalid mapping format: {e}")


class BaseDataset(Dataset):
    """
    Base dataset class for hierarchical multi-instance learning.
    
    This class provides basic functionality for loading and processing data
    in a hierarchical structure.
    """
    
    def __init__(self, split_data: List[Tuple[str, str, str]], cfg: DatasetConfig):
        """
        Initialize the base dataset.
        
        Args:
            split_data: List of (sample_id, label, path) tuples
            cfg: Dataset configuration object
        """
        self.list_sample = self.parse_input_list(split_data)
        self.logger = logging.getLogger(__name__)
        self.label_list = cfg.label_list
        self.mapping = cfg.mapping  # Now this is a dictionary
        
    def parse_input_list(self, split_data: List[Tuple[str, str, str]]) -> List[Tuple[str, str, str]]:
        """
        Parse and validate the input data list.
        
        Args:
            split_data: List of (sample_id, label, path) tuples
            
        Returns:
            Validated list of samples
            
        Raises:
            AssertionError: If the number of samples is 0
        """
        self.num_sample = len(split_data)
        assert self.num_sample > 0, "Dataset cannot be empty"
        self.logger.info(f'[dataset] # samples: {self.num_sample}')
        return split_data
        
    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return self.num_sample


class MILDataset(BaseDataset):


    def __init__(self, split_data: List[Tuple[str, str, str]], cfg: DatasetConfig):
        super().__init__(split_data, cfg)
        self.root_dataset = cfg.data_root

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:

        try:
            record_path = self.list_sample[index][2]

            record = torch.load(record_path)


            if record.dim() == 3:
                record = record.squeeze(0)


            label_name = self.list_sample[index][1]
            if label_name not in self.label_list:
                raise ValueError(f"Unknown label: {label_name}")
            label = self.label_list.index(label_name)

            return record, label

        except Exception as e:
            raise RuntimeError(f"Error loading sample {index}: {e}")