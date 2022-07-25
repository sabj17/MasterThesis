from encoders import BaseTransformer1
import torch
import torch.nn as nn
import torch.optim as optim
from frameworks import SupervisedTrain
import pytorch_lightning as pl
from data import BCI4, BNCI2014008, BNCI2014009, Physionet
from pytorch_lightning.loggers import WandbLogger
import wandb
from encoders import get_encoder
from embedders import get_embedder
from heads import TransformerClassificationHead
from utils import warmup_schedule
from einops.layers.torch import Reduce
import json
import os
from benchmark import get_dataset_from_subject, get_dataset_with_nsubs

import logging
logging.disable()


def get_dataset(dataset_name, subject, batch_size):
    batch_size = batch_size
    subjects = [subject]

    if dataset_name == 'bci4-2a':
        return BCI4(batch_size, subjects=subjects)
    elif dataset_name == 'physionet':
        return Physionet(batch_size, subjects=subjects)
    elif dataset_name == 'bci4-8':
        return BNCI2014008(batch_size, subjects=subjects)
    elif dataset_name == 'bci4-9':
        return BNCI2014009(batch_size, subjects=subjects)


def get_wandb_model(wandb_path, model_name, config, embed_size):
    wandb.restore(model_name, run_path=wandb_path)

    enc = get_encoder(config, embed_size)

    enc_state = torch.load(model_name)
    enc.load_state_dict(enc_state)

    os.remove(model_name)
    return enc


def set_trainable(model, trainable):
    for param in model.parameters():
        param.requires_grad = trainable


def train(model, trainloader, testloader, epochs, lr, warmup=0, logger=None):
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    scheduler = warmup_schedule(optimizer, warmup) if warmup > 0 else None
    process = SupervisedTrain(model, optimizer, scheduler)
    trainer = pl.Trainer(
        max_epochs=epochs, log_every_n_steps=10, gpus=1, logger=logger)
    trainer.fit(process, trainloader, testloader)

    if logger:
        logger.finalize('success')
        wandb.finish()


def finetune(wandb_path, model_name, dataset_name, n_classes, n_subs, batch_size=64):
    api = wandb.Api()
    run = api.run(wandb_path)
    config = json.loads(run.json_config)

    embed_size = config['embed_size']['value']

    run_name = run.name
    for subject in range(1, n_subs+1):

        dataset = get_dataset_from_subject(
            dataset_name, subject, batch_size=batch_size)
        trainloader, testloader, example, _ = dataset()

        enc = get_wandb_model(
            wandb_path, model_name, config['encoder']['value'], embed_size)

        model = nn.Sequential(
            get_embedder(config['embedder']['value'], example, embed_size),
            enc,
            TransformerClassificationHead(embed_size, n_classes)
        )

        # Bootstrap
        wandb_logger = WandbLogger(project='Model-eval', entity="eeg-masters",
                                   name=f'{run_name}-{dataset_name}-S{subject}-lineareval')
        set_trainable(enc, False)
        train(model, trainloader, testloader, 10, 1e-3, logger=wandb_logger)

        # Finetune
        wandb_logger = WandbLogger(project='Model-eval', entity="eeg-masters",
                                   name=f'{run_name}-{dataset_name}-S{subject}-finetune')
        set_trainable(enc, True)
        train(model, trainloader, testloader, 10, 3e-5, logger=wandb_logger)


def finetune_intersubject(project_name, wandb_path, model_name, dataset_name, n_classes, n_subs, batch_size=64):
    api = wandb.Api()
    run = api.run(wandb_path)
    config = json.loads(run.json_config)

    embed_size = config['embed_size']['value']

    run_name = run.name

    dataset = get_dataset_with_nsubs(
        dataset_name, n_subs, batch_size=batch_size)
    trainloader, testloader, example, _ = dataset()

    enc = get_wandb_model(
        wandb_path, model_name, config['encoder']['value'], embed_size)

    model = nn.Sequential(
        get_embedder(config['embedder']['value'], example, embed_size),
        enc,
        TransformerClassificationHead(embed_size, n_classes)
    )

    # Bootstrap
    wandb_logger = WandbLogger(project=project_name, entity="eeg-masters",
                               name=f'{run_name}-{dataset_name}-lineareval')
    set_trainable(enc, False)
    train(model, trainloader, testloader, 10, 1e-3, logger=wandb_logger)

    # Finetune
    wandb_logger = WandbLogger(project=project_name, entity="eeg-masters",
                               name=f'{run_name}-{dataset_name}-finetune')
    set_trainable(enc, True)
    train(model, trainloader, testloader, 10, 3e-5, logger=wandb_logger)


if __name__ == '__main__':
    finetune('Model-eval', 'eeg-masters/model-size/runs/nties5qp',
             'files/temp_11000.pt', 'bci4-2a', 9)
