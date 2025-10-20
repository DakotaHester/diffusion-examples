import torch
import cv2 as cv
from glob import glob
from torch.utils.data import Dataset
from torchvision.transforms import functional as F
from typing import Optional, Union, List, Tuple


class StandardDataAugmentations:
    '''
    Simple data augmentation that applies random rotation, horizontal, and vertical flips.
    '''
    
    @staticmethod
    def __call__(X: torch.Tensor, y: Optional[torch.Tensor] = None):
        
        # do not resize this time, just apply random flip and color distortions
        if torch.rand(1) > 0.5:
            X = F.hflip(X)
            if y is not None:
                y = F.hflip(y)
        
        if torch.rand(1) > 0.5:
            X = F.vflip(X)
            if y is not None:
                y = F.vflip(y)
        
        rot_angle = torch.randint(0, 4, (1,)).item()
        X = F.rotate(X, rot_angle * 90)
        if y is not None:
            y = F.rotate(y, rot_angle * 90)
    
        
        if y is not None:
            return X, y
        return X
    
class PlanetDataset(Dataset):
    
    def __init__(self, s2_filepaths: Union[List[str], Tuple[str]], ps_filepaths: Union[List[str], Tuple[str]], transforms: Optional[StandardDataAugmentations] = None):
        super().__init__()
        
        self.s2_filepaths = s2_filepaths
        self.ps_filepaths = ps_filepaths
        self.transforms = transforms
        
    def __len__(self):
        return len(self.s2_filepaths)
        
    @staticmethod
    def _scale(
        data, 
        in_range: Union[Tuple[int, int], Tuple[float, float]]=(0, 255), 
        out_range: Union[Tuple[int, int], Tuple[float, float]]=(-1.0, 1.0)
    ) -> torch.Tensor:
        
        # scale to 0-1
        data = (data - in_range[0]) / (in_range[1] - in_range[0])
        
        # scale to out_range
        data = data * (out_range[1] - out_range[0]) + out_range[0]
        data = data.clamp(min=out_range[0], max=out_range[1])
        return data
    
    def get_s2_img(self, idx):
        
        s2_img = cv.imread(self.s2_filepaths[idx])
        s2_img = cv.cvtColor(s2_img, cv.COLOR_BGR2RGB)
        
        s2_img = torch.as_tensor(s2_img, dtype=torch.float32)
        s2_img = s2_img.permute(2, 0, 1)
        return self._scale(s2_img)
    
    def get_ps_img(self, idx, harmonize: bool=False, return_s2_img: bool=False):
        
        if return_s2_img and not harmonize:
            raise ValueError('Cannot return Sentinel-2 image when harmonize is set to False')
        
        ps_img = cv.imread(self.ps_filepaths[idx])
        ps_img = cv.cvtColor(ps_img, cv.COLOR_BGR2RGB)

        ps_img = torch.as_tensor(ps_img, dtype=torch.float32)
        ps_img = self._scale(ps_img)
        ps_img = ps_img.permute(2, 0, 1)
        return ps_img
    
    
    def __getitem__(self, idx):
        # return self.get_ps_img(idx, harmonize=True, return_s2_img=True)
        s2_img, ps_img = self.get_s2_img(idx), self.get_ps_img(idx)
        if self.transforms is not None:
            return self.transforms(s2_img, ps_img)
        else:
            return s2_img, ps_img


# s2_image_paths = glob('/Volumes/dhester_ssd/dakota_sample_training_sr_images/*/*/s2_patch_*.png')
# ps_image_paths = glob('/Volumes/dhester_ssd/dakota_sample_training_sr_images/*/*/ps_patch_*.png')
s2_image_paths = glob('../dakota_sample_training_sr_images/*/*/s2_patch_*.png')
ps_image_paths = [fp.replace('s2_patch_', 'ps_patch_').replace('.png', '_harmonized.png') for fp in s2_image_paths]
