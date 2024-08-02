import torch
from torch import nn
from lightning import LightningModule


def kl_from_standard_normal(mean, log_var):
    kl = 0.5 * (log_var.exp() + mean.square() - 1.0 - log_var)
    return kl.mean()


def sample_from_standard_normal(mean, log_var, num=None):
    std = (0.5 * log_var).exp()
    shape = mean.shape
    if num is not None:
        # expand channel 1 to create several samples
        shape = shape[:1] + (num,) + shape[1:]
        mean = mean[:,None,...]
        std = std[:,None,...]
    return mean + std * torch.randn(shape, device=mean.device)


class AutoencoderKL(LightningModule):
    def __init__(
        self, 
        encoder, decoder, 
        kl_weight=0.01,     
        ae_flag=None,
        unet_regr=None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        # self.hidden_width = hidden_width
        self.encoded_channels = self.encoder.net[-1].out_channels
        self.to_moments = nn.Conv2d(self.encoded_channels, self.encoded_channels,
            kernel_size=1)
        self.to_decoder = nn.Conv2d(self.encoded_channels//2, self.encoded_channels,
            kernel_size=1)
        # self.log_var = nn.Parameter(torch.zeros(size=()))
        self.kl_weight = kl_weight
        self.ae_flag = ae_flag
        assert self.ae_flag in [None, 'residual', 'hres'], f'ae_flag {self.ae_flag} not recognized!!'
        if self.ae_flag=='residual':
            assert unet_regr is not None, 'If you want to work with residuals, provide a unet_regression network!'
        if unet_regr is not None:
            self.unet_regr = unet_regr.requires_grad_(False)
            self.unet_regr.eval()

    def encode(self, x):
        h = self.encoder(x)
        # print(f'H SIZE: {h.shape}')
        (mean, log_var) = torch.chunk(self.to_moments(h), 2, dim=1)
        # print(f'MEAN SIZE: {mean.shape}')
        return (mean, log_var)

    def decode(self, z):
        z = self.to_decoder(z)
        dec = self.decoder(z)
        return dec

    def forward(self, input, sample_posterior=True):
        (mean, log_var) = self.encode(input)
        if sample_posterior:
            z = sample_from_standard_normal(mean, log_var)
        else:
            z = mean
        # print(f'BOTTLENECK SIZE: {z.shape}')
        dec = self.decode(z)
        return (dec, mean, log_var)

    def _loss(self, batch):
        # (low_res, high_res, ts) = batch
        x,y = self.preprocess_batch(batch)
        while isinstance(x, list) or isinstance(x, tuple):
            x = x[0][0]
        (y_pred, mean, log_var) = self.forward(x)

        rec_loss = (y-y_pred).abs().mean()
        kl_loss = kl_from_standard_normal(mean, log_var)
        total_loss = rec_loss + self.kl_weight * kl_loss

        return (total_loss, rec_loss, kl_loss)

    def training_step(self, batch, batch_idx):
        loss = self._loss(batch)[0]
        self.log("train/train_loss", loss, sync_dist=True)
        return loss

    @torch.no_grad()
    def val_test_step(self, batch, batch_idx, split="val"):
        (total_loss, rec_loss, kl_loss) = self._loss(batch)
        log_params = {"on_step": False, "on_epoch": True, "prog_bar": True}
        self.log(f"{split}/loss", total_loss, **log_params, sync_dist=True)
        self.log(f"{split}/rec_loss", rec_loss.mean(), **log_params, sync_dist=True)
        self.log(f"{split}/kl_loss", kl_loss, **log_params, sync_dist=True)

    def validation_step(self, batch, batch_idx):
        self.val_test_step(batch, batch_idx, split="val")

    def test_step(self, batch, batch_idx):
        self.val_test_step(batch, batch_idx, split="test")

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-3,
            betas=(0.5, 0.9), weight_decay=1e-3)
        reduce_lr = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=3, factor=0.25, verbose=True
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": reduce_lr,
                "monitor": "val/rec_loss",
                "frequency": 1,
            },
        }
    
    def preprocess_batch(self, batch):
        (low_res, high_res, smt) = batch
        if self.ae_flag is None:
            return low_res, high_res
        elif self.ae_flag == 'residual':
            # Check sizes:
            if low_res.shape[-2::] != high_res.shape[-2::]:
                # Assuming you are running an ldm and therefore passing low_res not interpolated
                # and smt is static data: nearest-neighboring low-res and cat it to static data...')
                low_res = self.nn_lr_and_merge_with_static(low_res, smt)
            residual = high_res - self.unet(low_res)
            return residual, residual
        elif self.ae_flag == 'hres':
            return high_res, high_res
        else:
            print(f'ae_flag {self.ae_flag} not recognized!!')
            raise

    def nn_lr_and_merge_with_static(self, low_res, static_data):
        low_res = torch.repeat_interleave(low_res, 8, dim=2)
        low_res = torch.repeat_interleave(low_res, 8, dim=3)
        low_res = torch.cat([low_res, static_data], dim=1)
        return low_res

    @torch.no_grad()
    def unet(self, x):
        return self.unet_regr(x)


class EncoderLRES(LightningModule):
    def __init__(
        self, 
        encoder = nn.Identity(),   
        encoded_channels = 28,
        # hidden_width=8,
        # ckpt_path: str = None,
        # ignore_keys=[],
        **kwargs
    ):
        super().__init__(**kwargs)
        self.encoder = encoder
        # self.hidden_width = hidden_width
        self.encoded_channels = encoded_channels

    def encode(self, x):
        h = self.encoder(x)
        return (h,)
    
    def forward(self, x):
        h = self.encode()
        return (h,)
