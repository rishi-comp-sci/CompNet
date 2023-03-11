import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from compnet_data import compnet_dataset
from clip_unet import UNet
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ExponentialLR, ReduceLROnPlateau
    
class CompNet(pl.LightningModule):
    def __init__(self, data_path, save_path, batch_size, logger, workers=20):
        super(CompNet, self).__init__()
        
        self.lr = 0.0001
        self.batch_size = batch_size
        self.data_path = data_path
        self.workers = workers
        print("Data path: ", data_path)
        self.save_path = save_path
        PARAMS = {'batch_size': self.batch_size,
                  'init_lr': self.lr,
                  'data_path': self.data_path,
                  'save_path': self.save_path,
                    'scheduler': "Plateau"}
        # logger.log_hyperparams(PARAMS)
        
        self.unet = UNet()

        # Define beta schedule
        self.T = 1000
        self.betas = self.linear_beta_schedule(timesteps=self.T)

        # Pre-calculate different terms for closed form
        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)
        self.posterior_variance = self.betas * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)

    def forward(self, images, t, clip_encodings):
        pred_noise = self.unet(images, t, clip_encodings)
        return pred_noise
    
    def train_dataloader(self):
        train_dataset = compnet_dataset(root=self.data_path+"/train/", phase='train')
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, num_workers=self.workers, shuffle=True)
        # print("train_dataloader", len(train_loader))
        self.logger.log_hyperparams({'Num_train_files': len(train_dataset)})
        return train_loader
    
    def val_dataloader(self):
        dataVal = compnet_dataset(root=self.data_path+"/val/", phase='val')
        val_loader = DataLoader(dataVal, batch_size=self.batch_size, num_workers=self.workers, shuffle=False)
        # print("val_dataloader", len(dataVal))
        self.logger.log_hyperparams({'Num_val_files': len(dataVal)})
        return val_loader
    
    def training_step(self, batch, batch_idx):
        # print("TRAIN STEP!")
        image, t, clip_enc = batch
        x = self.forward_diffusion_sample(image, t)
        x = torch.dequantize(torch.quantize_per_tensor(torch.nn.functional.interpolate(torch.nn.functional.interpolate(x, size=64, mode='bicubic'), size=256, mode='bicubic'), 0.1, 10, torch.qint8))
        y = self.get_previous(image, t)
        y_hat = self(x, t, clip_enc)
        loss = F.mse_loss(x-y_hat, y)
        self.log('train_loss', loss, on_step=True, on_epoch=True)
        self.logger.experiment.log_metric('train_loss', loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        # print("VAL STEP!")
        image, t, clip_enc = batch
        x = self.forward_diffusion_sample(image, t)
        y = self.get_previous(image, t)
        y_hat = self(x, t, clip_enc)
        val_loss = F.mse_loss(x-y_hat, y)
        self.logger.experiment.log_metric('val_loss', val_loss)
        self.log('val_loss', val_loss, on_step=True, on_epoch=True)
        return val_loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, betas=(0.9, 0.999), eps=1e-07)
#         scheduler = ExponentialLR(optimizer, gamma=0.64, verbose=True)
        scheduler = ReduceLROnPlateau(optimizer, 'min', verbose=True)
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_loss'
            }
        }


    def linear_beta_schedule(self, timesteps, start=0.0001, end=0.02):
        return torch.linspace(start, end, timesteps)

    def get_index_from_list(self, vals, t, x_shape):
        """
        Returns a specific index t of a passed list of values vals
        while considering the batch dimension.
        """
        batch_size = t.shape[0]
        out = vals.gather(-1, t.cpu().type(torch.int64))
        return out.reshape(batch_size, *((1,)*(len(x_shape)-1))).to(t.device)

    def forward_diffusion_sample(self, x_0, t, device='cuda'):
        """ 
        Takes an image and a timestep as input and 
        returns the noisy version of it
        """
        noise = torch.randn_like(x_0)
        sqrt_alphas_cumprod_t = self.get_index_from_list(self.sqrt_alphas_cumprod, t, x_0.shape)
        sqrt_one_minus_alphas_cumprod_t = self.get_index_from_list(
            self.sqrt_one_minus_alphas_cumprod, t, x_0.shape
        )
        # mean + variance
        noisy_x = torch.tensor(sqrt_alphas_cumprod_t.to(device) * x_0.to(device) \
        + sqrt_one_minus_alphas_cumprod_t.to(device) * noise.to(device)) #, torch.from_numpy(noise.to(device))
        return noisy_x

    def get_previous(self, x_0, t):
        pt = (t - 1).cpu().numpy().flatten()
        y = []
        for i in range(len(pt)):
            if pt[i] <= 0:
                y.append(x_0[0].unsqueeze(dim=0))
            else: y.append(self.forward_diffusion_sample(x_0[0].unsqueeze(dim=0), torch.tensor([pt[i]-1.0])))
        return torch.cat(y)
