import numpy as np
import pandas as pd
import pickle
import os
import yaml
from scipy import signal
import wandb
from pytorch_lightning import loggers as pl_loggers
from gwpy.timeseries import TimeSeries

np.random.seed(42)


# Measure the average power spectral (asd) density of the noise targets
# The asd will be used for data whitening
def avg_psd_calculate(X_train, y_train, tukey_alpha):
    N_SAMPLES = 800
    idxs = {'noise': np.random.permutation(np.where(y_train == 0)[0])[0:N_SAMPLES],
            'sig': np.random.permutation(np.where(y_train == 1)[0])[0:N_SAMPLES],
            }

    # High-pass Filter each signal
    Nt = len(X_train[0][0])
    freqs = np.fft.rfftfreq(Nt, 1 / 2048)
    psds = np.empty((N_SAMPLES, 3, len(freqs)))

    # Tukey Window
    leakage_window = signal.tukey(4096, tukey_alpha)

    # Loop over the noise targets to measure the asd
    for i, idx in enumerate(idxs['noise']):
        sigs = X_train[idx]
        for j in range(3):
            sig = TimeSeries(sigs[j], sample_rate=2048)
            sig = sig * leakage_window
            psds[i, j] = sig.asd().value
    psd_avg = psds.mean(axis=0)
    Pxx = {'noise': dict(pxx=psds, psd_avg=psd_avg)}
    del X_train, y_train
    return freqs, Pxx['noise']['psd_avg']


# Return the list of configuration (cfg) files for bash execution
def list_of_cfgs():
    list_files = []
    for (dirpath, dirnames, filenames) in os.walk(f'./Model_Cfgs'):
        list_files += [os.path.join(dirpath, file) for file in filenames]
    list_files.sort()
    cfgs = []
    for file_name in list_files:
        # Read YAML file
        with open(file_name, 'r') as stream:
            data_loaded = yaml.safe_load(stream)
            cfgs.append([file_name, data_loaded])
    return cfgs


# Load the parameters from a configuration file
def load_cfg(file_name):
    file_path = os.path.join(f'./Model_Cfgs', file_name)
    # Read YAML file
    with open(file_path, 'r') as stream:
        data_loaded = yaml.safe_load(stream)
    return data_loaded


# Get the path of a checkpoint's weight for loading a model
def get_ckpt_path(name, id, ckpt_type, *, existing=False):
    """ Lookup Checkpoint Path """
    if existing:
        log_path = os.path.join('gwave', id, 'checkpoints')
    else:
        log_path = os.path.join(name, id, 'checkpoints')

    if ckpt_type == 'last':
        ckpt_file_name = [file for file in os.listdir(log_path) if 'last' in file]
    else:
        ckpt_file_name = [file for file in os.listdir(log_path) if ('last' not in file) and ('step' in file)]
    ckpt_file_name = ckpt_file_name[0]

    if existing:
        ckpt_path = {'dir': log_path,
                     'file_name': ckpt_file_name,
                     'path': os.path.join(log_path, ckpt_file_name)}
        print(f'Retrieved Existing *.ckpt: {os.path.join(log_path, ckpt_file_name)}')
    else:
        ckpt_path = {'dir': os.path.join(name, id, 'checkpoints'),
                     'file_name': ckpt_file_name,
                     'path': os.path.join(log_path, ckpt_file_name)}

    return ckpt_path


# Load the raw data from disk into RAM
class LoadData:
    def __init__(self, dataset_name, data_type, *, partial=False):
        self.dataset_name = dataset_name
        self.data_type = data_type
        self.partial = partial

        if partial:
            data_path = f'/media/dunlap/D65AEC245AEBFF5D/GravWave/{dataset_name}/{data_type}_signals_partial.npy'
        else:
            data_path = f'/media/dunlap/D65AEC245AEBFF5D/GravWave/{dataset_name}/{data_type}_signals.npy'
        self.data_path = data_path
        self.data_list_path = f'/media/dunlap/D65AEC245AEBFF5D/GravWave/{data_type}_files.pkl'

        stats = {
            'train': {
                'mean': np.load(f'/media/dunlap/D65AEC245AEBFF5D/GravWave/{self.dataset_name}/mean_train_signals.npy'),
                'std': np.load(f'/media/dunlap/D65AEC245AEBFF5D/GravWave/{self.dataset_name}/std_train_signals.npy')},
            'test': {
                'mean': np.load(f'/media/dunlap/D65AEC245AEBFF5D/GravWave/{self.dataset_name}/mean_test_signals.npy'),
                'std': np.load(f'/media/dunlap/D65AEC245AEBFF5D/GravWave/{self.dataset_name}/std_test_signals.npy')},
        }
        stats['norm'] = {'mean': np.array([stats['train']['mean'].mean(axis=0),
                                           stats['test']['mean'].mean(axis=0)]).mean(axis=0),
                         'std': np.array([stats['train']['std'].mean(axis=0),
                                          stats['test']['std'].mean(axis=0)]).mean(axis=0),
                         }
        self.stats = stats

    def load_data(self):
        # Load Numpy Data
        self.X = np.load(self.data_path)
        self.N = self.X.shape[0]

        # Load list of data directories from data
        with open(self.data_list_path, 'rb') as f:
            list_files = pickle.load(f)
        list_files = [i for i in list_files[0:self.N]]

        # Split data directory to get data ids
        ids = [i.split('/')[-1].split('.')[0] for i in list_files]

        # Place file paths and ids into dictionary
        self.data_info = {'paths': list_files,
                          'id': ids}

        # Return train or test data
        if self.data_type == 'train':
            train_df = pd.read_csv('./Data/g2net-gravitational-wave-detection/training_labels.csv')
            y = train_df.target.to_numpy()
            self.y = y[0:self.N]


# Data preprocessing with various signal processing filters, scaling, cross-correlation, etc.
class SigFilter:
    def __init__(self, *, fmin=15, fmax=1015, tukey_alpha=0.2, leakage_window_type='tukey', xcorr=False):
        self.fmin = fmin
        self.fmax = fmax
        self.fs = 2048
        self.leakage_window_type = leakage_window_type
        if leakage_window_type == 'tukey':
            self.leakage_window = signal.tukey(4096, tukey_alpha)
        elif leakage_window_type == 'boxcar':
            self.leakage_window = signal.boxcar(4096)
        self.xcorr = xcorr

    def __list_sigs_to_np(self, sigs_list):
        sigs = np.empty((3, sigs_list[0].shape[0]))
        for i in range(3):
            sigs[i] = sigs_list[i].value
        return sigs

    def __xcorr_sigs(self, sigs_list):
        sig_xcorr_list = []
        for i in range(3):
            if i == 0:
                sig_A = sigs_list[0]
                sig_B = sigs_list[1]
            if i == 1:
                sig_A = sigs_list[0]
                sig_B = sigs_list[2]
            if i == 2:
                sig_A = sigs_list[1]
                sig_B = sigs_list[2]
            sig_xcorr_list.append(sig_A.correlate(sig_B))
        sigs_xcorr = self.__list_sigs_to_np(sig_xcorr_list)
        return sigs_xcorr

    def filter_sigs(self, sigs):
        sigs_list = []
        for i in range(3):
            sig = TimeSeries(sigs[i], sample_rate=self.fs)
            if self.leakage_window_type == 'tukey':
                sig = sig * self.leakage_window
                sig = sig.whiten(asd=self.psd[i], window='boxcar')
            sig = sig.bandpass(flow=self.fmin, fhigh=self.fmax, filtfilt=True)
            sigs_list.append(sig)

        if self.xcorr:
            sigs_xcorr = self.__xcorr_sigs(sigs_list)
            sigs = self.__list_sigs_to_np(sigs_list)
            sigs = np.vstack((sigs, sigs_xcorr))
        else:
            sigs = self.__list_sigs_to_np(sigs_list)

        return sigs


# Measure various descriptive statistics about the data for EDA and normalizations
def sig_stats(x, y, hpf):
    x_filt = hpf.filter_sigs(x[0])

    sig_mean = np.empty((x.shape[0], x_filt.shape[0]))
    sig_std = np.empty((x.shape[0], x_filt.shape[0]))
    sig_min = np.empty((x.shape[0], x_filt.shape[0]))
    sig_max = np.empty((x.shape[0], x_filt.shape[0]))
    for i in range(x.shape[0]):
        sig_filt = hpf.filter_sigs(x[i])
        sig_mean[i] = sig_filt.mean(axis=1)
        sig_std[i] = sig_filt.std(axis=1)
        sig_min[i] = sig_filt.min(axis=1)
        sig_max[i] = sig_filt.max(axis=1)

    hpf.mean = sig_mean.mean(axis=0)
    hpf.std = sig_std.mean(axis=0)
    hpf.min = sig_min.mean(axis=0)
    hpf.max = sig_max.mean(axis=0)
    return hpf


# Log results for Weights & Biases
class WandB:
    def __init__(self, cfg):
        self.key = ''   # Enter your wandb key here
        self.id = wandb.util.generate_id()  # Generate version name for tracking in wandb
        self.wb_logger = pl_loggers.WandbLogger(project='gwave',
                                                config=cfg,
                                                name=self.id,
                                                version=self.id)
        wandb.login(key=self.key)
