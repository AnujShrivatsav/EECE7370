import torch
import numpy as np
import os.path as osp
import glob
import logging
from pathlib import Path
from torch.utils.data import Dataset
from datasets.dataset import DictDataset, DatasetPhase, str2datasetphase_type
from lib.utils import read_txt
from PIL import Image
import random

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

class ControlledEPNDataset(DictDataset):

    DATA_PATH_FILE = {
        DatasetPhase.Train: 'train',
        DatasetPhase.Test: 'test',
        DatasetPhase.Debug: 'test'
    }

    def __init__(self,
                 config,
                 input_transform=None,
                 target_transform=None,
                 augment_data=False,
                 cache=False,
                 phase=DatasetPhase.Train):

        if isinstance(phase, str):
            phase = str2datasetphase_type(phase)

        data_root = config.data.data_dir
        if config.data.per_class:
            data_paths = read_txt(osp.join(data_root, 'splits', self.DATA_PATH_FILE[phase]+'_'+config.data.class_id+'.txt'))
        else:
            pattern = osp.join(data_root, 'splits', f"{self.DATA_PATH_FILE[phase]}*.txt")
            txt_files = glob.glob(pattern)
            data_paths = []
            for txt_file in txt_files:
                data_paths.extend(read_txt(txt_file))
        data_paths = [data_path for data_path in data_paths]
        self.config = config
        self.representation = config.exp.representation
        self.trunc_distance = config.data.trunc_distance
        self.log_df = config.data.log_df
        self.data_paths = data_paths
        self.image_root = Path(config.data.image_dir)
        self.image_type = config.data.image_type
        self.augment_data = augment_data
        self.input_transform = input_transform
        self.target_transform = target_transform
        self.suffix = config.data.suffix
        # logging.info('Loading {}: {}'.format(self.__class__.__name__, self.DATA_PATH_FILE[phase]))

        DictDataset.__init__(
            self,
            data_paths,
            input_transform=input_transform,
            target_transform=target_transform,
            cache=cache,
            data_root=data_root)


    def get_output_id(self, iteration):
        return '_'.join(Path(self.data_paths[iteration]).stem.split('_')[:2])

    def load(self, filename):
        return torch.load(filename, weights_only=False)
    
    def load_image(self, filename):
        random_idx = np.random.randint(0, 5)
        image_idx = f"{random_idx:02d}"
        image_path = filename / f"{image_idx}.png"
        
        # Return None if image doesn't exist
        if not image_path.exists():
            return None
        
        return np.array(Image.open(image_path).convert('RGB'))

    def __len__(self):
        return len(self.data_paths)

    def __getitem__(self, index: int):
        while index < len(self.data_paths):
            filename = self.data_root / self.data_paths[index]
            image_rendering_path = self.image_root / self.data_paths[index].split('__')[0] / self.image_type
            rendered_image = self.load_image(image_rendering_path)
            
            # Skip this data point if image loading failed
            if rendered_image is None:
                index += 1
                continue
            
            scan_id = osp.basename(filename).replace(self.suffix, '')
            input_sdf, gt_df = self.load(filename)

            if self.representation == 'tsdf':
                input_sdf = np.clip(input_sdf, -self.trunc_distance, self.trunc_distance)
                gt_df = np.clip(gt_df, 0.0, self.trunc_distance)

            if self.log_df:
                gt_df = np.log(gt_df + 1)

            # Transformation
            if self.input_transform is not None:
                input_sdf = self.input_transform(input_sdf)
            if self.target_transform is not None:
                gt_df = self.target_transform(gt_df)

            return scan_id, rendered_image, input_sdf, gt_df
        
        raise IndexError("No valid data points found")
