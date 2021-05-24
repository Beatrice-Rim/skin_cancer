import random
import numpy as np
import argparse
import torch
import wandb
from torch.utils.data import random_split, DataLoader

import pytorch_lightning as pl
import skin_cancer


parser = argparse.ArgumentParser()
parser.add_argument('--train', action='store_true', default=False)
parser.add_argument('--do_test', action='store_true', default=False)
parser.add_argument('--test', action='store_true', default=False)
parser.add_argument('--n_epochs', type=int, default=1)
parser.add_argument('--use_gpu', action='store_true', default=False)
parser.add_argument('--model_type', type=str)
parser.add_argument('--dataset_root', type=str)
parser.add_argument('--checkpoint_path', type=str)


def fix_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    try:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except:
        pass


def train(model_type, dataset_root, n_epochs, use_gpu, do_test=True, batch_size=32, ckpt_metric='valid/loss'):
    wandb.login()
    experiment = wandb.init(project='skin_cancer', entity='bmipt', name=model_type, reinit=True)

    logger = pl.loggers.WandbLogger(experiment=experiment, log_model=True)

    model_checkpoint = pl.callbacks.ModelCheckpoint(
        dirpath=experiment.dir,
        filename=model_type,
        monitor=ckpt_metric
    )

    early_stopping = pl.callbacks.EarlyStopping(
        monitor='valid/loss',
        patience=10
    )

    trainer = pl.Trainer(
        logger=logger,
        callbacks=[model_checkpoint, early_stopping],
        gpus=int(use_gpu),
        max_epochs=n_epochs)

    if model_type == 'regression':
        model = skin_cancer.dl_models.RegressionModel(224*224*3, 1)
    elif model_type == 'mlp':
        model = skin_cancer.dl_models.MLP(224*224*3, 256, 1)
    elif model_type in ['resnet18', 'alexnet']:
        model = skin_cancer.dl_models.PretrainedModel(model_type, 1, pretrained=True)

    ds = skin_cancer.data.SkinDataset(dataset_root, split='train')
    n_samples = len(ds)
    train_size = int(0.7*n_samples)
    val_size = n_samples - train_size
    train_ds, val_ds = random_split(ds, (train_size, val_size), generator=torch.Generator().manual_seed(42))
    train_dl, val_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True), \
                       DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    module = skin_cancer.train.Model(
        model, torch.binary_cross_entropy_with_logits, lr=1e-3, class_to_idx=ds.class_to_idx)

    trainer.fit(module, train_dl, val_dl)
    experiment.finish()

    if do_test:
        test(model_checkpoint.best_model_path, dataset_root, use_gpu, trainer)
    return model_checkpoint.best_model_path


def test(checkpoint_path, dataset_root, use_gpu, trainer=None):
    test_ds = skin_cancer.data.SkinDataset(dataset_root, split='test')
    test_dl = DataLoader(test_ds, batch_size=32, shuffle=True)
    if trainer is None:
        trainer = pl.Trainer(gpus=int(use_gpu))
    trainer.test(ckpt_path=checkpoint_path, test_dataloaders=test_dl)


if __name__ == '__main__':
    fix_seed()
    args = parser.parse_args()
    if args.train:
        train(args.model_type, args.dataset_root, args.n_epochs, args.use_gpu, args.do_test)
    elif args.test:
        test(args.checkpoint_path, args.dataset_root, args.use_gpu)
