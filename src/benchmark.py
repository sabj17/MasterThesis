import torch
import torch.nn as nn
import torch.optim as optim
from frameworks import SupervisedTrain
import pytorch_lightning as pl
from data import BCI4, BNCI2014008, BNCI2014009, Physionet, SSVEPExo
import logging
from braindecode.models import EEGNetv4, EEGResNet, ShallowFBCSPNet, TCN, Deep4Net
from pytorch_lightning.loggers import WandbLogger
import wandb
logging.disable()


class ModelWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        x = self.model(x)
        if x.dim() == 3:
            x = torch.mean(x, dim=-1)

        return x


def get_dataset_from_subject(dataset_name, subject, batch_size=64):
    subjects = [subject]

    if dataset_name == 'bci4-2a':
        return BCI4(batch_size, subjects=subjects)
    elif dataset_name == 'physionet':
        return Physionet(batch_size, subjects=subjects)
    elif dataset_name == 'bci4-8':
        return BNCI2014008(batch_size, subjects=subjects)
    elif dataset_name == 'bci4-9':
        return BNCI2014009(batch_size, subjects=subjects)
    elif dataset_name == 'exo':
        return SSVEPExo(batch_size, subjects=subjects)


def get_dataset_with_nsubs(dataset_name, nsubs, batch_size=64):
    if dataset_name == 'bci4-2a':
        return BCI4(batch_size, n_subs=nsubs)
    elif dataset_name == 'physionet':
        return Physionet(batch_size, n_subs=nsubs)
    elif dataset_name == 'bci4-8':
        return BNCI2014008(batch_size, n_subs=nsubs)
    elif dataset_name == 'bci4-9':
        return BNCI2014009(batch_size, n_subs=nsubs)
    elif dataset_name == 'exo':
        return SSVEPExo(batch_size, n_subs=nsubs)


def get_model(model_name, n_channels, seq_len, n_classes):
    if model_name == 'EEGResNet':
        model = EEGResNet(n_channels, n_classes, seq_len,
                          final_pool_length='auto', n_first_filters=32)
    elif model_name == 'ShallowFBCSPNet':
        model = ShallowFBCSPNet(n_channels, n_classes,
                                seq_len, final_conv_length='auto')
    elif model_name == 'TCN':
        model = TCN(n_channels, n_classes, 4, 32, 3, 0.1, True)
    elif model_name == 'EEGNetv4':
        model = EEGNetv4(n_channels, n_classes, seq_len)
    elif model_name == 'Deep4Net':
        model = Deep4Net(n_channels, n_classes, seq_len,
                         final_conv_length='auto')

    return ModelWrapper(model)


def benchmark(project_name):
    datasets = [
        ('bci4-2a', 9, 4),
        ('physionet', 10, 5),  # subset of the 109
        ('bci4-8', 8, 2),
        ('bci4-9', 10, 2),
        ('exo', 12, 4)
    ]
    models = [
        'EEGResNet',
        'ShallowFBCSPNet',
        'TCN',
        'EEGNetv4',
        #  'Deep4Net'
    ]

    for dataset_name, n_subjects, n_classes in datasets:

        for subject_id in range(1, n_subjects+1):
            print(dataset_name, "- subject", subject_id)
            dataset = get_dataset_from_subject(dataset_name, subject_id)
            trainloader, testloader, example, _ = dataset()

            for model_name in models:
                _, n_channels, seq_len = example['tensor'].shape
                model = get_model(model_name, n_channels,
                                  seq_len, n_classes=n_classes)

                optimizer = optim.Adam(model.parameters())
                process = SupervisedTrain(
                    model, optimizer, n_classes=n_classes)

                wandb_logger = WandbLogger(
                    project=project_name, entity="eeg-masters", name=f'{dataset_name}-S{subject_id}-{model_name}')

                trainer = pl.Trainer(
                    max_epochs=10, log_every_n_steps=10, gpus=1, logger=wandb_logger)
                trainer.fit(process, trainloader, testloader)

                wandb_logger.finalize('success')
                wandb.finish()


def benchmark_inter_subject(project_name):
    datasets = [
        ('bci4-2a', 9, 4),
        ('physionet', 10, 5),  # subset of the 109
        ('bci4-8', 8, 2),
        ('bci4-9', 10, 2),
        ('exo', 24, 4)
    ]
    models = [
        'EEGResNet',
        'ShallowFBCSPNet',
        'TCN',
        'EEGNetv4',
        #  'Deep4Net'
    ]

    for dataset_name, n_subjects, n_classes in datasets:
        dataset = get_dataset_with_nsubs(
            dataset_name, n_subjects)
        trainloader, testloader, example, _ = dataset()

        for model_name in models:
            _, n_channels, seq_len = example['tensor'].shape
            model = get_model(model_name, n_channels,
                              seq_len, n_classes=n_classes)

            optimizer = optim.Adam(model.parameters())
            process = SupervisedTrain(
                model, optimizer, n_classes=n_classes)

            wandb_logger = WandbLogger(
                project=project_name, entity="eeg-masters", name=f'{dataset_name}-{model_name}')

            trainer = pl.Trainer(
                max_epochs=10, log_every_n_steps=10, gpus=1, logger=wandb_logger)
            trainer.fit(process, trainloader, testloader)

            wandb_logger.finalize('success')
            wandb.finish()


if __name__ == '__main__':
    benchmark('Benchmarks2')
