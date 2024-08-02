from typing import Any

import torch
from lightning import LightningModule


class UnetGANLitModule(LightningModule):

    def __init__(self,
        net: torch.nn.Module,
        loss: torch.nn.Module,
        base_learning_rate: float = 4.5e-6,
        ckpt_path: str = None,
        net_ckpt: torch.nn.Module = None,
        ignore_keys=[],
    ):
        super().__init__()
        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False, ignore=['loss'])
        self.automatic_optimization = False

        self.net = net
        self.loss = loss

        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)
        # self.lr_g_factor = lr_g_factor

    def init_from_ckpt(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location="cpu")["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        missing, unexpected = self.load_state_dict(sd, strict=False)
        print(f"Restored from {path} with {len(missing)} missing and {len(unexpected)} unexpected keys")
        if len(missing) > 0:
            print(f"Missing Keys: {missing}")
            print(f"Unexpected Keys: {unexpected}")
            
    def forward(self, x: torch.Tensor):
        return self.net(x)

    def training_step(self, batch, batch_idx):
        unet_opt, d_opt = self.optimizers()
        # https://github.com/pytorch/pytorch/issues/37142
        # try not to fool the heuristics
        lr, hr, ts_ns = batch
        hr_pred = self(lr)

        # unet
        optimizer_idx = 0
        self.toggle_optimizer(unet_opt)
        unetloss, log_dict_unet = self.loss(hr, hr_pred, optimizer_idx, self.global_step,
                                        last_layer=self.get_last_layer(), split="train")
        self.log("unetgan_loss", unetloss, prog_bar=True, logger=False, on_step=True, on_epoch=False, sync_dist=True)
        unet_opt.zero_grad()
        self.manual_backward(unetloss)
        unet_opt.step()
        self.untoggle_optimizer(unet_opt)

        # discriminator
        optimizer_idx = 1
        self.toggle_optimizer(d_opt)
        discloss, log_dict_disc = self.loss(hr, hr_pred, optimizer_idx, self.global_step,
                                        last_layer=self.get_last_layer(), split="train")
        self.log("disc_loss", discloss, prog_bar=True, logger=False, on_step=True, on_epoch=False, sync_dist=True)
        d_opt.zero_grad()
        self.manual_backward(discloss)
        d_opt.step()
        self.untoggle_optimizer(d_opt)

        self.log_dict({**log_dict_unet, **log_dict_disc}, prog_bar=False, logger=True, on_step=True, on_epoch=True)

    def validation_step(self, batch, batch_idx):
        log_dict = self._validation_step(batch, batch_idx)
        return log_dict

    def _validation_step(self, batch, batch_idx, suffix=""):
        lr, hr, ts_ns = batch
        hr_pred = self(lr)
        unetloss, log_dict_unet = self.loss(hr, hr_pred, 0, self.global_step,
                                        last_layer=self.get_last_layer(), split="val"+suffix)
        discloss, log_dict_disc = self.loss(hr, hr_pred, 1, self.global_step,
                                        last_layer=self.get_last_layer(), split="val"+suffix)
        rec_loss = log_dict_unet[f"val{suffix}/rec_loss"]
        self.log(f"val{suffix}/rec_loss", rec_loss,
                   prog_bar=True, logger=True, on_step=False, on_epoch=True, sync_dist=True)
        self.log(f"val{suffix}/unetgan_loss", unetloss,
                   prog_bar=True, logger=True, on_step=False, on_epoch=True, sync_dist=True)
        del log_dict_unet[f"val{suffix}/rec_loss"]
        self.log_dict(log_dict_unet)
        self.log_dict(log_dict_disc)
        return self.log_dict

    def configure_optimizers(self):
        bs = self.trainer.datamodule.hparams.batch_size
        agb = self.trainer.accumulate_grad_batches
        ngpu = self.trainer.num_devices
        # model.learning_rate = accumulate_grad_batches * ngpu * bs * base_lr
        # print(agb, ngpu, bs, self.base_learning_rate)
        self.learning_rate = agb * ngpu * bs * self.hparams.base_learning_rate
        unet_opt = torch.optim.Adam(self.net.parameters(),
                                  lr=self.learning_rate, betas=(0.5, 0.9), foreach=True)
        disc_opt = torch.optim.Adam(self.loss.discriminator.parameters(),
                                    lr=self.learning_rate, betas=(0.5, 0.9), foreach=True)

        return [unet_opt, disc_opt], []

    def get_last_layer(self):
        # defined the right layer
        return self.net.last_layer().weight
    
    def test_step(self, batch: Any, batch_idx: int):
        log_dict = self._test_step(batch, batch_idx)
        return log_dict

    def _test_step(self, batch, batch_idx, suffix=""):
        lr, hr, ts_ns = batch
        hr_pred = self(lr)
        unetloss, log_dict_unet = self.loss(hr, hr_pred, 0, self.global_step,
                                        last_layer=self.get_last_layer(), split="test"+suffix)
        discloss, log_dict_disc = self.loss(hr, hr_pred, 1, self.global_step,
                                        last_layer=self.get_last_layer(), split="test"+suffix)
        rec_loss = log_dict_unet[f"test{suffix}/rec_loss"]
        self.log(f"test{suffix}/rec_loss", rec_loss,
                   prog_bar=True, logger=True, on_step=False, on_epoch=True, sync_dist=True)
        self.log(f"test{suffix}/unetgan_loss", unetloss,
                   prog_bar=True, logger=True, on_step=False, on_epoch=True, sync_dist=True)
        del log_dict_unet[f"test{suffix}/rec_loss"]
        self.log_dict(log_dict_unet)
        self.log_dict(log_dict_disc)
        return self.log_dict

    def on_test_epoch_end(self):
        pass