import os
import yaml
from pathlib import Path
from abc import ABC, abstractmethod
from torch.utils.data import DataLoader
from data import get_data_info
from embedders import get_embedder
from encoders import get_encoder
from masking import get_masker
from frameworks import get_framework


class ConfigReader():
    def __init__(self):

        cwd = os.getcwd()
        file_path = cwd + '/experiments.yaml'
        self.doc = None
        with open(file_path) as file:
            self.doc = yaml.full_load(file)

    @abstractmethod
    def read(self):
        pass


class ModelConfig(ConfigReader):

    def read(self):
        baseline = self.doc['Baseline']
        """
        dataset = baseline['dataset']
        framework = baseline['framework']
        embed_size = baseline['embed_size']
        embedder = baseline['embedder']
        encoder = baseline['encoder']
        head = baseline['head']
        optimizer = baseline['optimizer']
        scheduler = baseline['scheduler']
        """

        # (dataset, framework, embed_size, embedder, encoder, head, optimizer, scheduler)
        return baseline


class ExperimentsConfig(ConfigReader):

    def read(self):
        return self.doc['Experiments']


def prepare_experiment(config):
    trainloader, testloader, data_exp, n_subjects = get_data_info(
        config['dataset'])
    embedding = get_embedder(
        config['embedder'], data_exp, config['embed_size'])
    encoder = get_encoder(config['encoder'], config["embed_size"])
    mask = get_masker(config['masker'], config['embed_size'])
    framework = get_framework(config['framework'], config['embed_size'],
                              embedding, encoder, mask, config['head'], data_exp, n_subjects)
    return framework, trainloader, testloader


def prepare_data(config):
    trainloader, testloader, data_exp, n_subjects = get_data_info(
        config['dataset'])
    return trainloader, testloader, data_exp, n_subjects


def prepare_framework(config, data_exp, n_subjects):
    embedding = get_embedder(
        config['embedder'], data_exp, config['embed_size'])
    encoder = get_encoder(config['encoder'], config["embed_size"])
    mask = get_masker(config['masker'], config['embed_size'])
    framework = get_framework(config['framework'], config['embed_size'],
                              embedding, encoder, mask, config['head'], data_exp, n_subjects)

    return framework
