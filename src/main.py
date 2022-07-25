import yaml
from config import *
import torch.optim as optim
from frameworks import PLWrapper
import pytorch_lightning as pl
import torch
import torch.nn as nn
import os
from pathlib import Path
from data import BCI4
from frameworks import SupervisedTrain
from pytorch_lightning.loggers import WandbLogger
import wandb
from embedders import PatchEmbeddings, get_embedder, EmbeddingWithSubject
import logging
from utils import LARS, warmup_schedule
logging.disable()


def get_optim(model, conf):
    params = conf['parameters']

    lr = getattr(params, 'lr', 0.001)
    weight_decay = getattr(params, 'weight_decay', 0.0)
    warmup_steps = getattr(params, 'warmup_steps', 0)

    scheduler = None
    if conf['type'] == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=lr,
                               weight_decay=weight_decay, betas=(0.5, 0.99))
    elif conf['type'] == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=lr,
                                weight_decay=weight_decay, betas=(0.5, 0.99))
    elif conf['type'] == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=lr,
                              weight_decay=weight_decay)
    elif conf['type'] == 'lars':
        optimizer = LARS(model.parameters(), lr=lr,
                         weight_decay=weight_decay)

    if warmup_steps > 0:
        scheduler = warmup_schedule(optimizer, warmup_steps)

    return optimizer, scheduler


def pretrain(experiment_name, experiment_file_name):
    """Main function"""
    cwd = Path(os.getcwd()).parent

    gpus = 0
    if torch.cuda.is_available():
        gpus = 1

    file_path = os.path.join(cwd, experiment_file_name)
    for i, config in enumerate(experiment_iter(file_path)):

        # Train self-supervised
        if i == 0:
            pretrainloader, _, pretrain_example, pretrain_n_sub = prepare_data(
                config)

        model = prepare_framework(
            config, pretrain_example, pretrain_n_sub)

        wandb_logger = WandbLogger(
            project=experiment_name, entity="eeg-masters", name=config['name'])
        wandb_logger.watch(model, log_freq=100, log="all")
        wandb_logger.experiment.config.update(config)

        optimizer, scheduler = get_optim(model, config['pretrain_optim'])

        process = PLWrapper(model, optimizer, scheduler)

        trainer = pl.Trainer(
            max_steps=config['pretrain_steps'], log_every_n_steps=1, gpus=gpus, logger=wandb_logger, gradient_clip_val=0.5)
        trainer.fit(process, pretrainloader)

        # Transfer learn
        if i == 0:
            dataset = BCI4(32, subjects=None)
            trainloader, testloader, example, _ = dataset()

        new_embedder = get_embedder(
            config['embedder'], example, config['embed_size'])
        new_model = model.get_frozen_enc_model2(
            new_embedder, 4, hidden_factor=1)

        optimizer, scheduler = get_optim(new_model, config['finetune_optim'])

        use_subject = isinstance(
            model.embedder, EmbeddingWithSubject)
        transfer_learn(new_model, trainloader, testloader, optimizer, scheduler,
                       config['finetune_steps'], wandb_logger, use_subject)

        wandb_logger.finalize('success')
        wandb.finish()


def transfer_learn(model, trainloader, testloader, optimizer, scheduler, steps, logger, use_subject):
    process = SupervisedTrain(
        model, optimizer, scheduler, use_subject=use_subject)

    trainer = pl.Trainer(
        max_steps=steps, log_every_n_steps=10, gpus=1, logger=logger, val_check_interval=500)
    trainer.fit(process, trainloader, testloader)

    torch.save(model.state_dict(), 'finetuned.h5')
    wandb.save('finetuned.h5')


def experiment_iter(file_name):
    with open(file_name) as config_file:
        experiment_dict = yaml.load(config_file.read(), Loader=yaml.FullLoader)

    config = experiment_dict['Baseline']
    experiments = experiment_dict['Experiments']

    for experiment in experiments:
        name, experiment = experiment.popitem()
        experiment_config = config.copy()
        experiment_config.update(experiment)

        yield experiment_config


if __name__ == '__main__':
    pretrain('test_tueg', 'experiments.yaml')
