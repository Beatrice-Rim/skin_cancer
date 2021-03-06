import pytorch_lightning as pl
import torch
import torchmetrics
import wandb


class Model(pl.LightningModule):
    def __init__(self, model, criterion, lr, class_to_idx):
        super(Model, self).__init__()
        self.model = model
        self.lr = lr
        self.criterion = criterion

        self.train_acc = torchmetrics.Accuracy()
        self.val_acc = torchmetrics.Accuracy()
        self.test_acc = torchmetrics.Accuracy()

        self.train_spec = torchmetrics.Specificity()
        self.val_spec = torchmetrics.Specificity()
        self.test_spec = torchmetrics.Specificity()

        self.train_sens = torchmetrics.Recall()
        self.val_sens = torchmetrics.Recall()
        self.test_sens = torchmetrics.Recall()

        self.class_to_idx = class_to_idx
        self.idx_to_class = {v: k for k, v in class_to_idx.items()} if class_to_idx is not None else None

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y = y.float()
        logits = self.model(x)
        pred = (logits > 0).long()
        loss = self.criterion(logits, y.float())

        y = y.long()
        self.log('train/accuracy', self.train_acc(pred, y), on_step=True, on_epoch=True)
        self.log('train/specificity', self.train_spec(pred, y), on_step=True, on_epoch=True)
        self.log('train/sensitivity', self.train_sens(pred, y), on_step=True, on_epoch=True)
        self.log('train/loss', loss, on_step=True, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self.model(x)
        loss = self.criterion(logits, y.float())
        pred = (logits > 0).long()
        y = y.long()
        self.log('valid/accuracy', self.val_acc(pred, y), on_step=False, on_epoch=True)
        self.log('valid/specificity', self.val_spec(pred, y), on_step=False, on_epoch=True)
        self.log('valid/sensitivity', self.val_sens(pred, y), on_step=False, on_epoch=True)
        self.log('valid/loss', loss, on_step=False, on_epoch=True)

        if batch_idx == 0 and isinstance(self.logger, pl.loggers.WandbLogger):
            self.logger.experiment.log(
                {'valid/images':
                    [
                        wandb.Image(x_,
                                    caption='' if self.idx_to_class is None else f'{self.idx_to_class[y_.item()]} - {self.idx_to_class[p_.item()]}')
                        for x_, y_, p_ in zip(x, y, pred)
                    ]
                }
            )

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self.model(x)
        loss = self.criterion(logits, y.float())
        pred = (logits > 0).long()
        y = y.long()
        self.log('test/accuracy', self.test_acc(pred, y), on_step=False, on_epoch=True)
        self.log('test/specificity', self.test_spec(pred, y), on_step=False, on_epoch=True)
        self.log('test/sensitivity', self.test_sens(pred, y), on_step=False, on_epoch=True)
        self.log('test/loss', loss, on_step=False, on_epoch=True)

        if batch_idx == 0 and isinstance(self.logger, pl.loggers.WandbLogger):
            self.logger.experiment.log(
                {'test/images':
                    [
                        wandb.Image(x_,
                                    caption='' if self.idx_to_class is None else f'{self.idx_to_class[y_.item()]} - {self.idx_to_class[p_.item()]}')
                        for x_, y_, p_ in zip(x, y, pred)
                    ]
                }
            )
