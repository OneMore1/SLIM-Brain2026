import os
import glob
import re 
import numpy as np
import pandas as pd 
import torch
from torch.utils.data import Dataset
from typing import List, Tuple, Union, Literal
import torch.nn.functional as F
from .pretrain_dataset import fMRIDataset
import io  
import nibabel as nib

class fMRITaskDataset(fMRIDataset):

    def __init__(
        self,
        data_root: str,
        datasets: List[str],
        split_suffixes: List[str],
        crop_length: int,
        label_csv_path: str,
        task_type: Literal['classification', 'regression'] = 'classification',
        downstream=True,
    ):
        super().__init__(data_root, datasets, split_suffixes, crop_length, downstream)
        
        self.task_type = task_type
        self.labels_map = self._load_and_process_labels(label_csv_path)

        initial_file_count = len(self.file_paths)
        self.file_paths = [
            path for path in self.file_paths 
            if self._extract_subject_id(path) in self.labels_map
        ]
        
        if len(self.file_paths) < initial_file_count:
            print(f"Warning: Dropped {initial_file_count - len(self.file_paths)} files due to missing labels in CSV.")
        
        print(f"Task Dataset ready for {self.task_type}. Usable files: {len(self.file_paths)}")


    def _extract_subject_id(self, file_path: str) -> str:


            # folder_name = os.path.basename(os.path.dirname(file_path))
            # match = re.search(r'(\d{7})', folder_name)

            match = re.search(r'(\d{6})', os.path.basename(file_path))
            
            if match:
                subject_id_with_zeros = match.group(1)
                subject_id = subject_id_with_zeros.lstrip('0') 
                
                return subject_id
                
            return "" 

    def _load_and_process_labels(self, csv_path: str) -> dict:

        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Label CSV file not found at: {csv_path}")
            
        print(f"Loading labels from {csv_path}...")
        df = pd.read_csv(csv_path)
        
        df['Subject'] = df['Subject'].astype(str)
        df.dropna(subset=['Subject'], inplace=True) 

        labels_map = {}
        
        if self.task_type == 'classification':
            label_col = None
            if 'Gender' in df.columns:
                label_col = 'Gender'
            elif 'gender' in df.columns:
                label_col = 'gender'
            elif 'age_group' in df.columns: 
                label_col = 'age_group'
            
            if label_col is None:
                raise ValueError("CSV must contain 'sex', 'gender' or 'age_group' column for classification.")

            print(f"Using column '{label_col}' as label.")
              
            # unique_vals = df[label_col].unique() 

            sex_mapping = {'F': 0, 'M': 1, 'f': 0, 'm': 1}
            
            if df[label_col].dtype == object and df[label_col].astype(str).iloc[0].upper() in ['F', 'M']:
                print(f"Encoding {label_col} (F/M) to Integers (0/1)...")
                df = df[df[label_col].isin(sex_mapping.keys())]
                df[label_col] = df[label_col].map(sex_mapping)
            else:
                df[label_col] = pd.to_numeric(df[label_col], errors='coerce').astype(int)
            
            for _, row in df.iterrows():
                subject_id = row['Subject']
                labels_map[subject_id] = torch.tensor(row[label_col], dtype=torch.long)

        elif self.task_type == 'regression':
            label_col = 'age'
            if label_col not in df.columns:
                 raise ValueError(f"Regression task requires '{label_col}' column.")
            df[label_col] = pd.to_numeric(df[label_col], errors='coerce')
            df.dropna(subset=[label_col], inplace=True)
            
            for _, row in df.iterrows():
                subject_id = row['Subject']
                labels_map[subject_id] = torch.tensor(row[label_col], dtype=torch.float32).view(1)

        else:
            raise ValueError(f"Unsupported task_type: {self.task_type}")

        print(f"Successfully loaded {len(labels_map)} subjects' labels.")
        return labels_map

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:

        retries = 0
        max_retries = 100 
        while retries < max_retries:
            try:
                data_tensor = super().__getitem__(idx)

                if data_tensor is None:
                    raise ValueError(f"Failed to load data at index {idx} (super returned None)")

                file_path = self.file_paths[idx]
                
                subject_id = self._extract_subject_id(file_path)

                data_tensor = data_tensor.unsqueeze(0)
                
                if subject_id in self.labels_map:
                    label_tensor = self.labels_map[subject_id]

                    return data_tensor, label_tensor
                else:
                    raise KeyError(f"Label not found for subject ID: {subject_id}")

            except Exception as e:
                # print(f"Warning: Error loading index {idx}: {e}. Retrying...")
                
                idx = np.random.randint(0, len(self))
                retries += 1
        
        raise RuntimeError(f"Failed to load any valid data after {max_retries} retries.")
            
        return data_tensor, label_tensor
