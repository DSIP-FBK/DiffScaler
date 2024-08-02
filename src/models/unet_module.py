from typing import Any

import torch
from lightning import LightningModule


class UnetLitModule(LightningModule):
    """Example of LightningModule for MNIST classification.

    A LightningModule organizes your PyTorch code into 6 sections:
        - Initialization (__init__)
        - Train Loop (training_step)
        - Validation loop (validation_step)
        - Test loop (test_step)
        - Prediction Loop (predict_step)
        - Optimizers and LR Schedulers (configure_optimizers)

    Docs:
        https://lightning.ai/docs/pytorch/latest/common/lightning_module.html
    """

    def __init__(
        self,
        net: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        loss: torch.nn.Module,
        scheduler: torch.optim.lr_scheduler = None,
        ckpt_path: str = None,
        ignore_keys=[],
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False, ignore=['net', 'loss'])

        self.net = net
        # loss function
        self.loss = loss

        # metric objects for calculating and averaging accuracy across batches
        # self.train_acc = Accuracy(task="multiclass", num_classes=10)
        # self.val_acc = Accuracy(task="multiclass", num_classes=10)
        # self.test_acc = Accuracy(task="multiclass", num_classes=10)

        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)

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
        # x = torch.repeat_interleave(x, 8, dim=2)
        # x = torch.repeat_interleave(x, 8, dim=3)
        return self.net(x)

    # def on_train_start(self):
    #     # by default lightning executes validation step sanity checks before training starts,
    #     # so it's worth to make sure validation metrics don't store results from these checks
    #     # self.val_loss.reset()

    def model_step(self, batch: Any):
        lr, hr, ts_ns = batch
        hr_pred = self.forward(lr)
        loss = self.loss(hr_pred, hr)
        return loss, hr_pred

    def training_step(self, batch: Any, batch_idx: int):
        loss, _ = self.model_step(batch)
        # update and log metrics
        # self.train_loss(loss)
        self.log("train/loss", loss.item(), on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        # return loss or backpropagation will fail
        return loss

    def on_train_epoch_end(self):
        pass

    def validation_step(self, batch: Any, batch_idx: int):
        loss, _ = self.model_step(batch)

        # update and log metrics
        # self.val_loss(loss)
        self.log("val/loss", loss.item(), on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)


    def on_validation_epoch_end(self):
        # acc = self.val_acc.compute()  # get current val acc
        # self.val_acc_best(acc)  # update best so far val acc
        # # log `val_acc_best` as a value through `.compute()` method, instead of as a metric object
        # # otherwise metric would be reset by lightning after each epoch
        # self.log("val/acc_best", self.val_acc_best.compute(), sync_dist=True, prog_bar=True)
        pass

    def test_step(self, batch: Any, batch_idx: int):
        loss, _ = self.model_step(batch)
        # update and log metrics
        # self.test_loss(loss)
        # self.test_acc(preds, targets)
        self.log("test/loss", loss.item(), on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

    def on_test_epoch_end(self):
        pass

    def configure_optimizers(self):
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://lightning.ai/docs/pytorch/latest/common/lightning_module.html#configure-optimizers
        """
        optimizer = self.hparams.optimizer(params=self.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}

