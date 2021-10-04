import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import os
from pytorch_lightning.core.lightning import LightningModule
import pytorch_lightning as pl
import torchmetrics
from pytorch_lightning import seed_everything
from audiomentations import Compose, Gain, Shift, TimeStretch, PitchShift, PolarityInversion
import torch.nn as nn

seed_everything(42)
np.random.seed(42)


def norm_signals(x, hp):
    x_norm = np.empty(x.shape)
    for i in range(x.shape[0]):
        if i < 3:
            x_norm[i] = x[i] / hp.max[i]
        else:
            x_norm[i] = (x[i] - hp.mean[i]) / hp.std[i]
    return x_norm


def augment_waves(aug):
    aug_features = []

    if aug['PolarityInversion']['apply']:
        aug_feature = PolarityInversion(p=aug['PolarityInversion']['prob'])
        aug_features.append(aug_feature)
        print('Polarity Inversion was Selected')

    if aug['TimeStretch']['apply']:
        aug_feature = TimeStretch(min_rate=aug['TimeStretch']['min_rate'],
                                  max_rate=aug['TimeStretch']['max_rate'],
                                  p=aug['TimeStretch']['prob'])
        aug_features.append(aug_feature)
        print('Time Stretch was Selected')

    if aug['PitchShift']['apply']:
        aug_feature = PitchShift(min_semitones=aug['PitchShift']['min_semitones'],
                                 max_semitones=aug['PitchShift']['max_semitones'],
                                 p=aug['PitchShift']['prob'])
        aug_features.append(aug_feature)

    if aug['Shift']['apply']:
        aug_feature = Shift(min_fraction=-1.0 * aug['Shift']['fraction'],
                            max_fraction=aug['Shift']['fraction'],
                            p=aug['Shift']['prob'])
        aug_features.append(aug_feature)

    if aug['Gain']['apply']:
        aug_feature = Gain(min_gain_in_db=aug['Gain']['min_gain_in_db'],
                           max_gain_in_db=aug['Gain']['max_gain_in_db'],
                           p=aug['Gain']['prob'])
        aug_features.append(aug_feature)

    if len(aug_features) == 0:
        augment = None
    else:
        augment = Compose(aug_features)
    return augment


# Training and Validation Dataset Loader
class GwTrainDataset(Dataset):
    def __init__(self, x, y, data_type, hpf, aug):
        self.x = x
        self.y = y
        self.data_type = data_type
        self.hpf = hpf
        self.aug = aug

    # Return length of dataset
    def __len__(self):
        return len(self.y)

    # Return (feature, label) pair
    def __getitem__(self, idx):
        # Single instance of data
        x_ = self.x[idx]
        if self.data_type == 'train' and self.aug is not None:
            x_ = self.aug(x_, sample_rate=2048)
        x_ = self.hpf.filter_sigs(x_)
        x_ = norm_signals(x_, self.hpf)
        x_ = torch.tensor(x_, dtype=torch.float32)
        y_ = torch.tensor(self.y[idx])
        return (x_, y_)


# Test Dataset Loader
class GwTestDataset(Dataset):
    def __init__(self, x, df_idxs, hpf, invert):
        self.x = x
        self.df_idxs = df_idxs
        self.hpf = hpf
        if invert:
            self.aug = Compose([PolarityInversion(p=1.0)])
        else:
            self.aug = Compose([PolarityInversion(p=0.0)])

    # Return length of dataset
    def __len__(self):
        return len(self.x)

    # Return (feature) only
    def __getitem__(self, idx):
        # Single instance of data
        x_ = self.x[idx]
        x_ = self.aug(x_, sample_rate=2048)
        x_ = self.hpf.filter_sigs(x_)
        x_ = norm_signals(x_, self.hpf)
        x_ = torch.tensor(x_, dtype=torch.float32)
        df_idxs_ = self.df_idxs[idx]
        return (x_, df_idxs_)


# Lightning Datamodule
class GwDataModule(pl.LightningDataModule):
    def __init__(self, x_train, y_train, x_val, y_val, *, batch_size, hpf, aug_info):
        super().__init__()
        self.x_train = x_train
        self.y_train = y_train
        self.x_val = x_val
        self.y_val = y_val
        self.batch_size = batch_size
        self.hpf = hpf

        # Get Signal augmentations
        self.aug = augment_waves(aug_info)

    # Setup
    def setup(self, stage=None):
        # Setup Training Data
        self.train_data = GwTrainDataset(x=self.x_train,
                                         y=self.y_train,
                                         data_type='train',
                                         hpf=self.hpf,
                                         aug=self.aug,
                                         )

        # Setup Validation data
        self.val_data = GwTrainDataset(x=self.x_val,
                                       y=self.y_val,
                                       data_type='val',
                                       hpf=self.hpf,
                                       aug=self.aug,
                                       )

    # Train DataLoader
    def train_dataloader(self):
        return DataLoader(self.train_data,
                          batch_size=self.batch_size,
                          shuffle=True,
                          num_workers=8,
                          pin_memory=True,
                          )

    # Validation DataLoader
    def val_dataloader(self):
        return DataLoader(self.val_data,
                          batch_size=self.batch_size,
                          num_workers=8,
                          pin_memory=True,
                          )


# Neural Network Model
class GwModel(LightningModule):

    def __init__(self, *, model_inputs, lr_inputs, batch_size, loss_fn, sig_inputs):
        super().__init__()
        self.model_name = model_inputs['name']
        self.scheduler_inputs = lr_inputs
        self.batch_size = batch_size
        self.loss_fn = loss_fn
        self.sig_inputs = sig_inputs
        self.save_hyperparameters()

        if loss_fn == 'bce':
            self.criterion = nn.BCEWithLogitsLoss()

        # Metrics
        self.train_auroc = torchmetrics.AUROC(num_classes=None)
        self.val_auroc = torchmetrics.AUROC(num_classes=None)
        #     # Get a backbone model

        if sig_inputs['xcorr']:
            in_channels = 6
        else:
            in_channels = 3

        fac = 3
        fac_kernel = 1

        self.cnn1 = nn.Sequential(
            nn.Conv1d(in_channels, int(64 * fac), kernel_size=64 * fac_kernel),
            nn.BatchNorm1d(int(64 * fac)),
            nn.SiLU(),
        )
        self.cnn2 = nn.Sequential(
            nn.Conv1d(int(64 * fac), int(64 * fac), kernel_size=32 * fac_kernel),
            nn.AvgPool1d(kernel_size=8),
            nn.BatchNorm1d(int(64 * fac)),
            nn.SiLU(),
        )
        self.cnn3 = nn.Sequential(
            nn.Conv1d(int(64 * fac), int(128 * fac), kernel_size=32 * fac_kernel),
            nn.BatchNorm1d(int(128 * fac)),
            nn.SiLU(),
        )
        self.cnn4 = nn.Sequential(
            nn.Conv1d(int(128 * fac), int(128 * fac), kernel_size=16 * fac_kernel),
            nn.AvgPool1d(kernel_size=6),
            nn.BatchNorm1d(int(128 * fac)),
            nn.SiLU(),
        )
        self.cnn5 = nn.Sequential(
            nn.Conv1d(int(128 * fac), int(256 * fac), kernel_size=16 * fac_kernel),
            nn.BatchNorm1d(int(256 * fac)),
            nn.SiLU(),
        )
        self.cnn6 = nn.Sequential(
            nn.Conv1d(int(256 * fac), int(256 * fac), kernel_size=16 * fac_kernel),
            nn.AvgPool1d(kernel_size=4),
            nn.BatchNorm1d(int(256 * fac)),
            nn.SiLU(),
        )
        self.fc1 = nn.Sequential(
            nn.Linear(int((256 * fac)) * 11, 768),
            nn.BatchNorm1d(768),
            nn.Dropout(0.5),
            nn.SiLU(),
        )
        self.fc2 = nn.Sequential(
            nn.Linear(768, 128),
            nn.BatchNorm1d(128),
            nn.Dropout(0.50),
            nn.SiLU(),
        )
        self.fc3 = nn.Sequential(
            nn.Linear(128, 1),
        )

    def forward(self, x, pos=None):
        x = self.cnn1(x)
        x = self.cnn2(x)
        x = self.cnn3(x)
        x = self.cnn4(x)
        x = self.cnn5(x)
        x = self.cnn6(x)
        x = x.flatten(start_dim=1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x

    def training_step(self, batch, batch_idx):
        data, target = batch
        output = torch.squeeze(self.forward(data))
        loss = self.criterion(output, target.float())
        self.train_auroc.update(output.sigmoid(), target)
        self.log('train/loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        data, target = batch
        output = torch.squeeze(self.forward(data))
        val_loss = self.criterion(output, target.float())
        self.val_auroc.update(output.sigmoid(), target)
        self.log('val/loss', val_loss, on_step=False, on_epoch=True, prog_bar=True)
        return {'val_loss': val_loss}

    def configure_optimizers(self):
        # Unpack learning rate scheduler inputs
        T_0 = self.scheduler_inputs['T_0']
        T_mult = self.scheduler_inputs['T_mult']
        lr_initial = self.scheduler_inputs['lr_initial']
        lr_min = self.scheduler_inputs['lr_min']
        optimizer = torch.optim.Adam(self.parameters(), lr=lr_initial)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,
                                                                         T_0=T_0,
                                                                         T_mult=T_mult,
                                                                         eta_min=lr_min,
                                                                         last_epoch=-1,
                                                                         )
        return [optimizer], [scheduler]

    def training_epoch_end(self, _):
        self.log('train/auroc', self.train_auroc.compute(), on_epoch=True, prog_bar=True)
        self.train_auroc.reset()

    def validation_epoch_end(self, _):
        self.log('val/auroc', self.val_auroc.compute(), on_epoch=True, prog_bar=True)
        self.val_auroc.reset()

    def on_save_checkpoint(self, checkpoint):
        checkpoint['ckpt_metrics'] = self.trainer.logged_metrics


# Define Checkpoint Save Path
def define_ckpt_save_directory(log_path):
    all_subdirs = [os.path.join(log_path, d) for d in os.listdir(log_path)]
    latest_subdir = max(all_subdirs, key=os.path.getmtime)
    new_subdir = latest_subdir.split('_')[0] + '_' + str(int(latest_subdir.split('_')[1]) + 1)
    return new_subdir


# Lookup Checkpoint Path
def get_ckpt_path(name, id, ckpt_type):
    log_path = os.path.join(name, id, 'checkpoints')
    if ckpt_type == 'last':
        ckpt_file_name = [file for file in os.listdir(log_path) if 'last' in file]
    else:
        ckpt_file_name = [file for file in os.listdir(log_path) if 'last' not in file]
    ckpt_file_name = ckpt_file_name[0]

    ckpt_path = {'dir': os.path.join(name, id, 'checkpoints'),
                 'file_name': ckpt_file_name,
                 'path': os.path.join(log_path, ckpt_file_name)}
    return ckpt_path
