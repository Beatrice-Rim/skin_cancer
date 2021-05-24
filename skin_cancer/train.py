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

        self.train_acc = torchmetrics.Accuracy(threshold=0)
        self.val_acc = torchmetrics.Accuracy(threshold=0)

        self.train_spec = torchmetrics.Specificity(threshold=0)
        self.val_spec = torchmetrics.Specificity(threshold=0)

        self.train_sens = torchmetrics.Recall(threshold=0)
        self.val_sens = torchmetrics.Recall(threshold=0)

        self.class_to_idx = class_to_idx
        self.idx_to_class = {v: k for k, v in class_to_idx.items()} if class_to_idx is not None else None

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self.model(x)
        loss = self.criterion(logits, y)

        self.log('train/accuracy', self.train_acc(logits, y), on_step=True, on_epoch=True)
        self.log('train/specificity', self.train_spec(logits, y), on_step=True, on_epoch=True)
        self.log('train/sensitivity', self.train_sens(logits, y), on_step=True, on_epoch=True)
        self.log('train/loss', loss, on_step=True, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self.model(x)
        loss = self.criterion(logits, y)
        pred = (logits > 0).long()

        self.log('valid/accuracy', self.valid_acc(logits, y), on_step=False, on_epoch=True)
        self.log('valid/specificity', self.valid_spec(logits, y), on_step=False, on_epoch=True)
        self.log('valid/sensitivity', self.valid_sens(logits, y), on_step=False, on_epoch=True)
        self.log('valid/loss', loss, on_step=False, on_epoch=True)

        if batch_idx == 0 and isinstance(self.logger, pl.loggers.WandbLogger):
            self.logger.experiment.log(
                {'valid/images':
                    [
                        wandb.Image(x_,
                                    caption='' if self.idx_to_class is None else f'{self.idx_to_class[y_]} - {self.idx_to_class[p_]}')
                        for x_, y_, p_ in zip(x, y, pred)
                    ]
                }
            )

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self.model(x)
        loss = self.criterion(logits, y)
        pred = (logits > 0).long()

        self.log('test/accuracy', self.valid_acc(logits, y), on_step=False, on_epoch=True)
        self.log('test/specificity', self.valid_spec(logits, y), on_step=False, on_epoch=True)
        self.log('test/sensitivity', self.valid_sens(logits, y), on_step=False, on_epoch=True)
        self.log('test/loss', loss, on_step=False, on_epoch=True)

        if batch_idx == 0 and isinstance(self.logger, pl.loggers.WandbLogger):
            self.logger.experiment.log(
                {'test/images':
                    [
                        wandb.Image(x_,
                                    caption='' if self.idx_to_class is None else f'{self.idx_to_class[y_]} - {self.idx_to_class[p_]}')
                        for x_, y_, p_ in zip(x, y, pred)
                    ]
                }
            )
