import torch
from braindecode.datasets import MOABBDataset, TUH
from braindecode.preprocessing import exponential_moving_standardize, preprocess, Preprocessor, scale
from braindecode.preprocessing import create_windows_from_events, create_fixed_length_windows
from braindecode.models import get_output_shape
import mne
from torch.utils.data import DataLoader

import contextlib
import sys
import os
import json


class TorchifiedDataset(torch.utils.data.Dataset):
    def __init__(self, bd_dataset):
        self.dataset = bd_dataset
        self.indexer = {}

        index = 0
        for i, ds in enumerate(self.dataset.datasets):
            for j in range(len(ds)):
                self.indexer[index] = (i, j)
                index += 1

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        ds_index, sample_index = self.indexer[idx]

        dataset = self.dataset.datasets[ds_index]
        subject = dataset.description['subject']

        sample = dataset[sample_index]

        X, y, _ = sample

        return X, y, subject - 1  # to start from zero


class TorchifiedDatasetTueg(torch.utils.data.Dataset):
    def __init__(self, bd_dataset):
        self.dataset = bd_dataset
        self.subject_info = self.get_subjects()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        ds_index = idx % len(self.dataset.datasets)

        dataset = self.dataset.datasets[ds_index-1]
        sample_index = idx % len(dataset)
        subject = dataset.description['subject']

        X, y, _ = dataset[sample_index]

        return X, y, self.get_subjects().get(str(subject))

    def get_subjects(self):

        #subject_path = "/content/drive/MyDrive/Master/Data/Tueg/download/subject_ids.txt"
        path = os.path.join(os.getcwd(), "subject_recordings.json")
        with open(path) as file:
            data = json.load(file)
        subject_idx = {}
        keys = list(data.keys())
        for idx in range(0, len(keys)):
            subject_idx.update({keys[idx]: idx})
        return subject_idx


def load_dataset(config, preload=False):
    dataset = MOABBDataset(
        dataset_name=config['dataset_name'], subject_ids=config['subjects'])
    n_subjects = len(dataset.description['subject'].unique())

    low_cut_hz = config['low_cut']
    high_cut_hz = config['high_cut']
    window_len_s = config['window_len']
    offset_s = config['offset']
    stride = config['stride']

    sfreq = 250

    preprocessors = [
        Preprocessor('pick_types', eeg=True, meg=False, stim=False),
        Preprocessor(scale, factor=1e6, apply_on_array=True),
        Preprocessor('filter', l_freq=low_cut_hz, h_freq=high_cut_hz),
        Preprocessor(exponential_moving_standardize,
                     factor_new=1e-3, init_block_size=250),
        Preprocessor('resample', sfreq=sfreq),
    ]

    preprocess(dataset, preprocessors)

    n_preds_per_input = None if window_len_s is None else stride
    input_window_samples = None if window_len_s is None else int(
        sfreq * window_len_s)

    trial_start_offset_samples = int(offset_s * sfreq)
    windows_dataset = create_windows_from_events(
        dataset,
        trial_start_offset_samples=trial_start_offset_samples,
        trial_stop_offset_samples=0,
        window_size_samples=input_window_samples,
        window_stride_samples=n_preds_per_input,
        drop_last_window=False,
        preload=preload
    )

    return windows_dataset, n_subjects

####### FACTORY STUFF #######


class BaseDataLoader:
    def __init__(self, batch_size, num_workers, n_subs):
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.n_subs = n_subs


class MoabbDataloader(BaseDataLoader):
    def __init__(self, config, batch_size, num_workers, n_subs, preload=False, subjects=None):
        super().__init__(batch_size, num_workers=num_workers, n_subs=n_subs)

        if subjects is None:
            config['subjects'] = list(range(1, self.n_subs+1))
        else:
            config['subjects'] = subjects

        self.dataset, self.n_subjects = load_dataset(config, preload=False)

    def __call__(self):
        dataset = TorchifiedDataset(self.dataset)

        train_size = int(0.5 * len(dataset))
        test_size = len(dataset) - train_size

        train_set, test_set = torch.utils.data.random_split(
            dataset, [train_size, test_size])

        trainloader = DataLoader(
            train_set, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)
        testloader = DataLoader(
            test_set, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

        example = next(iter(trainloader))
        tensor = example[0]
        output = len(example[1])

        return trainloader, testloader, {'tensor': tensor, 'output': output}, self.n_subjects


class BNCI2014008(MoabbDataloader):
    def __init__(self, batch_size, subjects=None, num_workers=0, n_subs=8):
        config = {
            'dataset_name': 'BNCI2014008',
            'low_cut': 0.1,
            'high_cut': 30.0,
            'window_len': None,
            'offset': 0.0,
            'stride': 5,
        }
        super().__init__(config, batch_size, num_workers,
                         n_subs, preload=False, subjects=subjects)


class BNCI2014009(MoabbDataloader):
    def __init__(self, batch_size, subjects=None, num_workers=0, n_subs=8):
        config = {
            'dataset_name': 'BNCI2014009',
            'low_cut': 0.1,
            'high_cut': 20.0,
            'window_len': 0.9,
            'offset': -0.1,
            'stride': 5,
        }
        super().__init__(config, batch_size, num_workers,
                         n_subs, preload=False, subjects=subjects)


class SSVEPExo(MoabbDataloader):
    def __init__(self, batch_size, subjects=None, num_workers=0, n_subs=8):
        config = {
            'dataset_name': 'SSVEPExo',
            'low_cut': 0.5,
            'high_cut': 100.0,
            'window_len': 1.9,
            'offset': -0.1,
            'stride': 5,
        }
        super().__init__(config, batch_size, num_workers,
                         n_subs, preload=False, subjects=subjects)


class Physionet(MoabbDataloader):
    def __init__(self, batch_size, subjects=None, num_workers=0, n_subs=8):
        config = {
            'dataset_name': 'PhysionetMI',
            'low_cut': 0.5,
            'high_cut': 79.0,
            'window_len': 4,
            'offset': -0.5,
            'stride': 5,
        }
        super().__init__(config, batch_size, num_workers,
                         n_subs, preload=False, subjects=subjects)


class BCI4(MoabbDataloader):
    def __init__(self, batch_size, subjects=None, num_workers=0, n_subs=9):
        config = {
            'dataset_name': 'BNCI2014001',
            'low_cut': 0.5,
            'high_cut': 100.0,
            'window_len': 4,
            'offset': -0.5,
            'stride': 10,
        }
        super().__init__(config, batch_size, num_workers,
                         n_subs, preload=False, subjects=subjects)

    def __call__(self):
        splitted = self.dataset.split('session')

        train_set = TorchifiedDataset(splitted['session_T'])
        test_set = TorchifiedDataset(splitted['session_E'])

        trainloader = DataLoader(
            train_set, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)
        testloader = DataLoader(
            test_set, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
        example = next(iter(trainloader))
        tensor = example[0]
        output = len(example[1])

        return trainloader, testloader, {'tensor': tensor, 'output': output}, self.n_subjects


def load_dataset_tueg(datapath, recording_ids, preload, window_len_s):
    mne.set_log_level('WARNING')

    ds = TUH(
        datapath, recording_ids=recording_ids, target_name=None,
        preload=preload)

    low_cut_hz = 4.0
    high_cut_hz = 38.0
    factor_new = 1e-3
    init_block_size = 1000

    preprocessors = [
        Preprocessor(scale, factor=1e6, apply_on_array=True),
        Preprocessor('filter', l_freq=low_cut_hz, h_freq=high_cut_hz),
        Preprocessor(exponential_moving_standardize,
                     factor_new=factor_new, init_block_size=init_block_size)
    ]

    preprocess(ds, preprocessors)

    fs = ds.datasets[0].raw.info['sfreq']
    # print(len(ds.datasets[0].raw.info['subject']))
    window_len_samples = int(fs * window_len_s)
    window_stride_samples = window_len_samples // 4
    windows_ds = create_fixed_length_windows(
        ds, start_offset_samples=0, stop_offset_samples=None,
        window_size_samples=window_len_samples,
        window_stride_samples=window_stride_samples, drop_last_window=True,
        preload=preload, drop_bad_windows=True)

    # Drop bad epochs
    # XXX: This could be parallelized.
    # XXX: Also, this could be implemented in the Dataset object itself.
    for ds in windows_ds.datasets:
        ds.windows.drop_bad()
        assert ds.windows.preload == preload

    return TorchifiedDatasetTueg(windows_ds)


def dataloader_tueg(preload=False, n_subjects=10, window_len_s=4, batch_size=64, num_workers=8, pin_memory=False):
    dataset = load_dataset_tueg(preload, window_len_s, n_subs=n_subjects)

    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, pin_memory=pin_memory,
        num_workers=num_workers, worker_init_fn=None)

    return dataloader, n_subjects


class TUEEG(BaseDataLoader):
    def __init__(self, batch_size, num_workers=0, n_subs=20):
        self.cwd = os.getcwd()
        #self.TUH_PATH = os.path.join(self.cwd, "download/v1.1.0/edf/01_tcp_ar")
        self.TUH_PATH = os.path.join(
            '/content/drive/MyDrive/Master/Data/Tueg', "download/v1.1.0/edf/01_tcp_ar")
        self.subject_path = self.cwd
        subs = n_subs
        #self.subject_recording_ids = self.get_datasets_from_subjectids()
        self.subject_recording_ids = self.get_datasets_from_subjectids()

        if subs is None:
            self.subs = len(self.subject_recording_ids)  # check
            self.subject_list = list(range(0, self.subs))
        elif type(subs) is list:
            self.subs = len(subs)
            self.subject_list = subs
        else:
            self.subs = subs
            self.subject_list = list(range(0, subs))
        super().__init__(batch_size, num_workers, subs)

    def __call__(self):

        train_idx = int(len(self.subject_list)*0.8)
        train_subjects = self.subject_list[0:train_idx-1]
        test_subjects = self.subject_list[train_idx:]

        train_recordings = []
        for i in train_subjects:
            key = list(self.subject_recording_ids.keys())[i]
            for record in self.subject_recording_ids.get(key):
                train_recordings.append(record)

        # print(train_recordings)

        test_recordings = []
        for i in test_subjects:
            key = list(self.subject_recording_ids.keys())[i]
            for record in self.subject_recording_ids.get(key):
                test_recordings.append(record)

        #train_recordings = [x for x in self.recording_ids.get(i) for i in train_subjects]
        #test_recordings = [x for x in self.recording_ids.get(i) for i in test_subjects]

        train_set = load_dataset_tueg(
            self.TUH_PATH, train_recordings, preload=False, window_len_s=4)
        # test_set = load_dataset_tueg(
        #    self.TUH_PATH, test_recordings, preload=False, window_len_s=4)

        trainloader = DataLoader(
            train_set, batch_size=self.batch_size, shuffle=True, num_workers=1)
        # testloader = DataLoader(
        #   test_set, batch_size=self.batch_size, shuffle=False, num_workers=1)

        print(len(trainloader))
        print(type(trainloader))

        example = next(iter(trainloader))
        tensor = example[0]
        output = len(example[1])

        return trainloader, None, {'tensor': tensor, 'output': output}, len(self.subject_recording_ids)

    def get_datasets_from_subjectids(self):
      # Read subject file: each line represents a subject with all related datasets

        subject_path = os.path.join(
            self.subject_path, "subject_recordings.json")
        with open(subject_path) as file:
            data = json.load(file)
        return data


def get_data_info(dataset_info):
    factories = {
        "bci4": BCI4,
        "tueeg": TUEEG,
        "p300-1": BNCI2014008,
        "p300-2": BNCI2014009,
        "physionet": Physionet,
        "exo": SSVEPExo
    }
    dataset_name = dataset_info['type']
    batch_size = dataset_info['batch_size']
    num_workers = dataset_info['num_workers']
    n_subjects = dataset_info['n_subjects']

    if dataset_name in factories:
        trainloader, testloader, example, n_subjects = factories[dataset_name](
            batch_size, num_workers=num_workers, n_subs=n_subjects)()
        return trainloader, testloader, example, n_subjects
    print(f"Unknown dataset option: {dataset_name}.")
