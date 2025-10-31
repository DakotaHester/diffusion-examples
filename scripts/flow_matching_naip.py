import torch
import torch.nn as nn
import torch.nn.functional as F
from glob import glob
from torch.utils.data import Dataset, DataLoader
from diffusers.models import UNet2DModel
from torchvision.transforms import functional as tvf
from thop import profile
import rasterio as rio
import os
import random
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
from torch import autocast
from torch.amp import GradScaler
from torchvision.models import vgg19, VGG19_Weights
from typing import Optional, Union, List, Tuple
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from joblib import dump, load
from torchmetrics import functional as tmf
import pandas as pd

def main():
    random.seed(1701)

    s2_files = glob(os.path.join('..', 's2_naip_pairs_4x', 'sentinel2', '*.tif'))
    lcmap_files = glob(os.path.join('..', 's2_naip_pairs_4x', 'lcmap', '*.tif'))
    naip_files = glob(os.path.join('..', 's2_naip_pairs_4x', 'naip', '*.tif'))

    s2_ids = [path.split(os.sep)[-1].replace('.tif', '') for path in s2_files]
    lcmap_ids = [path.split(os.sep)[-1].replace('.tif', '') for path in lcmap_files]
    naip_ids = [path.split(os.sep)[-1].replace('.tif', '') for path in naip_files]

    all_ids = [idfr for idfr in s2_ids if (idfr in lcmap_ids) and (idfr in naip_ids)]
    val_ids = random.sample(all_ids, k=int(0.1*len(all_ids)))
    train_ids = [idfr for idfr in all_ids if idfr not in val_ids]

    train_dataset = NAIPDataset(os.path.join('..', 's2_naip_pairs_4x'), ids=train_ids)
    print(f'Number of samples in training dataset: {len(train_dataset)}')

    val_dataset = NAIPDataset(os.path.join('..', 's2_naip_pairs_4x'), ids=val_ids)
    print(f'Number of samples in validation dataset: {len(val_dataset)}')

    pca_pipeline_path = os.path.join('..', 'models', 'pca_pipeline.joblib')
    if os.path.exists(pca_pipeline_path):
        pca_pipeline = load(pca_pipeline_path)
    else:
        X = np.concat([train_dataset.get_naip(i).numpy().transpose(1, 2, 0).reshape(-1, 4) for i in range(len(train_dataset))])
        pca_pipeline = make_pipeline(
            StandardScaler(),
            PCA(n_components=3)
        )
        pca_pipeline.fit(X)
        dump(pca_pipeline, pca_pipeline_path)

    # device = torch.device('mps') if torch.backends.mps.is_available() else torch.device('cpu')
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f'Using backed {device}')
    if device.type == 'cuda' and torch.cuda.is_bf16_supported():
        print('Using bfloat16 for mixed precision.')
        dtype = torch.bfloat16
    else:
        print('Using full precision (float32).')
        dtype = torch.float32

    model = UNet2DModel(
        sample_size=256,
        in_channels=16,
        out_channels=4,
        # block_out_channels=[96, 192, 384, 768],
        block_out_channels=[64, 128, 256, 512],
        # block_out_channels=[48, 96, 192, 384],
        # block_out_channels=[32, 32, 32, 32],
        # block_out_channels=[32, 64, 128, 256],
        down_block_types=['DownBlock2D'] * 4, # + ['AttnDownBlock2D'],
        up_block_types=['UpBlock2D'] * 4, # + ['AttnUpBlock2D']
        # norm_num_groups=48,
    )
    # # Gradient clipping
    # # See https://stackoverflow.com/a/54816498
    # clip_value = 1.0
    # for p in model.parameters():
    #     p.register_hook(lambda grad: torch.clamp(grad, -clip_value, clip_value))
    
    model.to(device, dtype=dtype)

    input = torch.randn(1, 16, 256, 256).to(device, dtype=dtype)
    macs, params = profile(model, inputs=(input,0), verbose=False)
    flops = 2 * macs 
    print(f"MACs: {macs:,}, Params: {params:,}, FLOPs: {flops:,}")

    batch_size = 256
    micro_batch_size = 32
    n_epochs = 300
    warmup_epochs = 10
    lr = 1e-3
    # loss_lambda = 0.0 # l_total = l_l1 + lambda * l_percep
    grad_accum_steps = batch_size // micro_batch_size

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, fused=torch.cuda.is_available())
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs-1)
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs - warmup_epochs)
    lr_scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warmup_epochs])
    # scaler = GradScaler()
    # perceptual_loss = PerceptualLoss(pca_pipeline, device=device)
    pca_conv = PCAConvLayer(pca_pipeline).to(device, dtype=dtype)

    train_dataloader = DataLoader(train_dataset, batch_size=micro_batch_size, shuffle=True, drop_last=True, pin_memory=True, num_workers=4)
    val_dataloader = DataLoader(val_dataset, batch_size=micro_batch_size, shuffle=False, drop_last=False, pin_memory=True, num_workers=4)

    metrics = {
        'epoch': [],
        'lr': [],
    }
    for phase in ('train', 'val'):
        for metric in ('l1_loss', 'ssim', 'mssim', 'psnr', 'lpips'):
            metrics[f'{phase}_{metric}'] = [] 

    for epoch in range(1, n_epochs+1):
        
        metrics['epoch'].append(epoch)
        metrics['lr'].append(optimizer.param_groups[0]['lr'])
        
        epoch_metrics = {
            f'{phase}_{metric}': 0.0
            for metric in ('l1_loss', 'ssim', 'mssim', 'psnr', 'lpips')
            for phase in ('train', 'val')
        }
        
        for phase in ('train', 'val'):
            if phase == 'train':
                loader = train_dataloader
                model.train()
                torch.set_grad_enabled(True)
                
            else:
                loader = val_dataloader
                model.eval()
                torch.set_grad_enabled(False)
                
            phase_count = 0
        
            with tqdm(loader, desc=f'Epoch {epoch}/{n_epochs} {phase.capitalize() + ('  ' if phase == 'val' else '')}', unit='batch', postfix={'lr': optimizer.param_groups[0]['lr']}) as pbar:
                for i, (s2_img, lcmap, naip_img) in enumerate(pbar):
                    
                    s2_img, _, naip_img = s2_img.to(device, dtype=dtype), lcmap.to(device, dtype=dtype), naip_img.to(device, dtype=dtype)
                    
                    x_0 = torch.randn_like(naip_img) 
                    t = torch.rand(x_0.shape[0], device=device, dtype=dtype)
                    t_reshaped = t.view(-1, 1, 1, 1)
                    
                    # interpolated point on the path (linear interpolation)
                    x_t = (1 - t_reshaped) * x_0 + t_reshaped * naip_img
                    target_vector = naip_img - x_0 # target vector field
                    unet_input = torch.cat((x_t, s2_img), dim=1)

                    pred_vector = model(unet_input, t * 1000).sample
                    l1_loss = F.l1_loss(pred_vector, target_vector, reduction='none').mean(dim=(1, 2, 3))
                    # percep_loss = perceptual_loss(pred_vector + x_0, naip_img)
                        
                    # total_loss = l1_loss + loss_lambda * percep_loss
                    pred_img = (pred_vector + x_0).detach().clamp(-1, 1)
                    
                    phase_count += s2_img.shape[0]
                    epoch_metrics[f'{phase}_l1_loss']     += l1_loss.detach().sum().item()
                    # epoch_metrics[f'{phase}_percep_loss'] += percep_loss.detach().sum().item()
                    # epoch_metrics[f'{phase}_total_loss']  += total_loss.detach().sum().item()
                    epoch_metrics[f'{phase}_psnr']        += tmf.image.peak_signal_noise_ratio(pred_img, naip_img.detach(), data_range=(-1, 1), reduction='none', dim=(1, 2, 3)).sum().item()
                    epoch_metrics[f'{phase}_ssim']        += tmf.image.structural_similarity_index_measure(pred_img, naip_img.detach(), data_range=(-1, 1), reduction='none').sum().item()
                    epoch_metrics[f'{phase}_mssim']       += tmf.image.multiscale_structural_similarity_index_measure(pred_img, naip_img.detach(), data_range=(-1, 1), reduction='none').sum().item()
                    epoch_metrics[f'{phase}_lpips']       += tmf.image.learned_perceptual_image_patch_similarity(pca_conv.to_lpips(pred_img), pca_conv.to_lpips(naip_img.detach()), reduction='none').sum().item()
                    # lpips expects inputs in range [-1, 1] with 3 channels, but PCA outputs mean 0, std 1. We scale by 3 to get 99.7% of values in [-1, 1]

                    if phase == 'train':
                        # scaler.scale(total_loss.mean()).backward()
                        # scaler.scale(l1_loss.mean()).backward()
                        (l1_loss / grad_accum_steps).mean().backward()
                        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        if (i + 1) % grad_accum_steps == 0:
                            # scaler.step(optimizer)
                            # scaler.update()
                            optimizer.step()
                            optimizer.zero_grad()
                    
                    pbar_postfix = {'lr': f'{optimizer.param_groups[0]["lr"]:.2e}'}
                    for metric in ('l1_loss', 'ssim', 'mssim', 'psnr', 'lpips'):
                        avg = epoch_metrics[f'{phase}_{metric}'] / phase_count
                        pbar_postfix[metric] = f'{avg:.2e}'
                    pbar.set_postfix(pbar_postfix)
                                                        
            for metric in ('l1_loss', 'ssim', 'mssim', 'psnr', 'lpips'):
                epoch_metrics[f'{phase}_{metric}'] /= phase_count
                metrics[f'{phase}_{metric}'].append(epoch_metrics[f'{phase}_{metric}'])
                    
        pd.DataFrame(metrics).to_csv('../logs/fm_s2_naip_sr_4x.csv', index=False)
        torch.save(model.state_dict(), '../models/fm_s2_naip_sr_4x.pt')
        lr_scheduler.step()


class StandardDataAugmentations:
    '''
    Simple data augmentation that applies random rotation, horizontal, and vertical flips.
    Applies the same random transforms to all input tensors.
    '''
    @staticmethod
    def __call__(*images: torch.Tensor):
        # Generate random transforms
        do_hflip = torch.rand(1) > 0.5
        do_vflip = torch.rand(1) > 0.5
        rot_angle = torch.randint(0, 4, (1,)).item() * 90

        transformed = []
        for img in images:
            if do_hflip:
                img = tvf.hflip(img)
            if do_vflip:
                img = tvf.vflip(img)
            img = tvf.rotate(img, rot_angle)
            transformed.append(img)
        return tuple(transformed)
 
    
class NAIPDataset(Dataset):
    
    def __init__(self, 
        # s2_filepaths: Union[List[str], Tuple[str]], 
        # ps_filepaths: Union[List[str], Tuple[str]], 
        path: str,
        ids: List[str],
        s2_subdir: str='sentinel2',
        lcmap_subdir: str='lcmap',
        naip_subdir: str='naip',
        transforms: Optional[StandardDataAugmentations] = None
    ):
        super().__init__()
        
        self.path = path
        self.ids = ids
        self.s2_subdir = s2_subdir
        self.lcmap_subdir = lcmap_subdir
        self.naip_subdir = naip_subdir
        self.transforms = transforms
        
    def __len__(self):
        return len(self.ids)
        
    @staticmethod
    def _scale(
        data, 
        in_range: Union[Tuple[int, int], Tuple[float, float]]=(0, 10000), 
        out_range: Union[Tuple[int, int], Tuple[float, float]]=(-1.0, 1.0)
    ) -> torch.Tensor:
        
        # scale to 0-1
        data = (data - in_range[0]) / (in_range[1] - in_range[0])
        
        # scale to out_range
        data = data * (out_range[1] - out_range[0]) + out_range[0]
        data = data.clamp(min=out_range[0], max=out_range[1])
        return data

    
    def get_naip(self, idx):
        
        naip = torch.as_tensor(rio.open(os.path.join(self.path, self.naip_subdir, self.ids[idx] + '.tif')).read())
        return self._scale(naip.float())
    
    
    def __getitem__(self, idx):
        # return self.get_ps_img(idx, harmonize=True, return_s2_img=True)
        # s2_img, ps_img = self.get_s2_img(idx), self.get_ps_img(idx)
        
        s2 = torch.as_tensor(rio.open(os.path.join(self.path, self.s2_subdir, self.ids[idx] + '.tif')).read())
        lcmap = torch.as_tensor(rio.open(os.path.join(self.path, self.lcmap_subdir, self.ids[idx] + '.tif')).read(1))
        naip = torch.as_tensor(rio.open(os.path.join(self.path, self.naip_subdir, self.ids[idx] + '.tif')).read())
        
        s2 = self._scale(s2.float())
        naip = self._scale(naip.float())
        
        label_map = {
            1: 0,   # Water
            2: 1,   # Trees
            4: 2,   # Flooded Vegetation
            5: 3,   # Crops
            7: 4,   # Built Area
            8: 5,   # Bare Ground
            9: 6,   # Snow/Ice
            11: 7,  # Rangeland
            # 10: clouds, will be all zeros
        }

        lcmap_mapped = torch.full_like(lcmap, fill_value=255)  # 255 = invalid

        for orig, mapped in label_map.items():
            lcmap_mapped[lcmap == orig] = mapped

        mask_clouds = (lcmap == 10) | (lcmap_mapped == 255)
        lcmap_mapped[mask_clouds] = 0
        lcmap_onehot = F.one_hot(lcmap_mapped.long(), num_classes=8)  # shape (H, W, 8)
        lcmap_onehot[mask_clouds] = 0
        lcmap_onehot = lcmap_onehot.permute(2, 0, 1).float()
        
        if self.transforms is not None:
            return self.transforms(s2, lcmap_onehot, naip)
        else:
            return s2, lcmap_onehot, naip


class PCAConvLayer(nn.Module):
    """
    1x1 conv that performs StandardScaler -> PCA projection (frozen).
    Provides helpers to scale the PCA channels to the range expected by LPIPS.
    """
    def __init__(self, pca_pipeline):
        super().__init__()
        
        mu = pca_pipeline['standardscaler'].mean_
        sigma = pca_pipeline['standardscaler'].scale_
        V_T = pca_pipeline['pca'].components_.T

        # weight/bias that map original 4-channel image -> PCA scores
        W = np.diag(1.0 / sigma) @ V_T
        b = - (mu / sigma) @ V_T

        self.pca_conv = nn.Conv2d(4, 3, kernel_size=1, stride=1, bias=True)

        weight_tensor = torch.from_numpy(W.T).float().unsqueeze(-1).unsqueeze(-1)
        bias_tensor = torch.from_numpy(b).float()

        with torch.no_grad():
            self.pca_conv.weight.copy_(weight_tensor)
            self.pca_conv.bias.copy_(bias_tensor)

        # freeze parameters
        for p in self.pca_conv.parameters():
            p.requires_grad = False

        # store per-component std (explained variance of PCA on standardized data)
        pca_var = pca_pipeline['pca'].explained_variance_  # shape (3,)
        pca_std = np.sqrt(pca_var).astype(np.float32)
        # register as buffer so it moves with the module and survives state_dict
        self.register_buffer('pca_std', torch.from_numpy(pca_std).view(1, 3, 1, 1))


    def forward(self, x):
        """Return raw PCA scores (B,3,H,W)."""
        return self.pca_conv(x)

    def to_lpips(self, x, k: Optional[float]=2.0, clamp: bool=True):
        """
        Convert original 4-ch input x -> scaled PCA image suitable for LPIPS.
        - k: number of stds that should map to 1.0 (default=2.0).
        Returns tensor in approximately [-1, 1] (clamped if clamp=True).
        """
        out = self.pca_conv(x)  # B,3,H,W
        std = self.pca_std.to(out.device).type_as(out)
        scaled = out / (k * std)
        return scaled.clamp(-1.0, 1.0) if clamp else scaled


class PerceptualLoss(nn.Module):
    
    def __init__(self, pca_pipeline=None, device='cpu', reduction='batch'):
        super().__init__()
        
        self.device = device
        self.reduction = reduction

        if pca_pipeline is not None:
            self.pca_conv = self._create_pca_conv_layer(pca_pipeline).to(self.device)
            self.pca_conv.requires_grad_(False)
        
        self.vgg_feature_extractor = vgg19(weights=VGG19_Weights.IMAGENET1K_V1).features.to(self.device).eval()
        for p in self.vgg_feature_extractor.parameters():
            p.requires_grad_(False)
        
        self.feature_layers = [2, 7, 16, 25, 34]
        self.layer_weights = [0.1, 0.1, 1.0, 1.0, 1.0]
                
        if self.reduction == 'batch':            
            self.mae_loss = nn.L1Loss(reduction='none')
        else:
            self.mae_loss = nn.L1Loss(reduction=self.reduction)

    @staticmethod
    def _create_pca_conv_layer(pca_pipeline) -> nn.Conv2d:
        """Creates a frozen 1x1 Conv2D layer from scaler and PCA stats."""
        mu = pca_pipeline['standardscaler'].mean_
        sigma = pca_pipeline['standardscaler'].scale_
        V_T = pca_pipeline['pca'].components_.T
        
        W = np.diag(1 / sigma) @ V_T
        b = - (mu / sigma) @ V_T
        
        pca_conv = nn.Conv2d(4, 3, kernel_size=1, stride=1, bias=True)
        
        weight_tensor = torch.from_numpy(W.T).float().unsqueeze(-1).unsqueeze(-1)
        bias_tensor = torch.from_numpy(b).float()
        
        with torch.no_grad():
            pca_conv.weight.copy_(weight_tensor)
            pca_conv.bias.copy_(bias_tensor)
            
        return pca_conv
    
    def extract_features(self, x):
        """Extract features from specified conv layers (pre-activation)."""
        features = []
        for i, layer in enumerate(self.vgg_feature_extractor):
            x = layer(x)
            if i in self.feature_layers:
                features.append(x)
        return features

    def forward(self, generated_img: torch.Tensor, target_img: torch.Tensor) -> torch.Tensor:
        """
        Computes the Real-ESRGAN style perceptual loss.
        """
        generated_img = generated_img.to(self.device)
        target_img = target_img.to(self.device)
        
        # Apply PCA conversion if needed
        if hasattr(self, 'pca_conv'):
            gen_pca = self.pca_conv(generated_img)
            tgt_pca = self.pca_conv(target_img)
        else:
            gen_pca, tgt_pca = generated_img, target_img
        
        # Extract VGG features
        gen_feats = self.extract_features(gen_pca)
        tgt_feats = self.extract_features(tgt_pca)
        
        # Compute weighted perceptual loss
        total_loss = 0.0
        for w, gf, tf in zip(self.layer_weights, gen_feats, tgt_feats):
            loss = self.mae_loss(gf, tf)
            if self.reduction == 'batch':
                loss = loss.mean(dim=(1, 2, 3))  # per-image mean
            total_loss += w * loss

        return total_loss


def inference(model, s2_img, lcmap, total_steps):
    
    device = model.device
    dtype = next(model.parameters()).dtype
    timesteps = torch.linspace(0, 1, total_steps, device=device)
    step_size = timesteps[1] - timesteps[0]
    
    naip_shape = list(s2_img.shape)
    naip_shape[1] = 4
    x = torch.randn(naip_shape, device=device, dtype=dtype)
    for t in tqdm(timesteps):
        t_batch = torch.ones(s2_img.shape[0], device=device, dtype=dtype) * t
        t_mid_batch = torch.ones(s2_img.shape[0], device=device, dtype=dtype) * (t + step_size / 2)
        t_next_batch = torch.ones(s2_img.shape[0], device=device, dtype=dtype) * (t + step_size)
            
        # k1: velocity at start of the step
        model_input_k1 = torch.cat((x, s2_img, lcmap), dim=1)
        v_k1 = model(model_input_k1, t_batch * 1000).sample
        
        # k2: velocity at midpoint
        x_k2 = x + v_k1 * (step_size / 2)
        model_input_k2 = torch.cat((x_k2, s2_img, lcmap), dim=1)
        v_k2 = model(model_input_k2, t_mid_batch * 1000).sample
        
        # k3: velocity at midpoint (estimated with k2)
        x_k3 = x + v_k2 * (step_size / 2)
        model_input_k3 = torch.cat((x_k3, s2_img, lcmap), dim=1)
        v_k3 = model(model_input_k3, t_mid_batch * 1000).sample
        
        # k4: velocity at the end of the step
        x_k4 = x + v_k3 * step_size
        model_input_k4 = torch.cat((x_k4, s2_img, lcmap), dim=1)
        v_k4 = model(model_input_k4, t_next_batch * 1000).sample
        
        x = x + (step_size / 6) * (v_k1 + 2*v_k2 + 2*v_k3 + v_k4)
    
    return x


if __name__ == '__main__':
    main()
