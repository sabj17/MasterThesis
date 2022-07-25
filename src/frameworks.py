import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from einops import reduce, rearrange, repeat
from einops.layers.torch import Rearrange, Reduce
from copy import deepcopy
from losses import BYOLLoss, NT_Xent
import torchmetrics
from heads import get_head, TransformerClassificationHead, FlattenHead


# temporary imports
from encoders import FNet, SubjectInputEncoder, MultiTokenOutEncoder
from utils import GradualEmaModel
from embedders import PatchEmbeddings, EmbeddingWithSubject
from masking import RandomTokenMasking
from losses import SmoothL1Loss
from tokenizers import EEGtoPatchTokens
from utils import Sequential
import wandb
import os


class BaseFrameWork(nn.Module):
    def __init__(self, params, modules, embed_size, n_subjects):
        super().__init__()
        self.params = params

        self.embed_size = embed_size
        self.n_subjects = n_subjects

        self.encoder = modules['encoder']
        self.head = modules['head']
        self.embedder = modules['embedding']
        self.masker = modules['mask']

    def param(self, name):
        assert name in self.params
        return self.params[name]

    def get_teacher_targets(self, teacher, x, k):
        with torch.no_grad():
            _, z = self.teacher(x)
            z = torch.mean(z[-k:], dim=0)
            z = z.detach().clone()

        return z

    def get_encoder(self):
        self.encoder.return_sides = False
        return Sequential(
            deepcopy(self.embedder),
            deepcopy(self.encoder)
        )

    def get_finetune_model(self, n_classes, hidden_factor=2):
        return Sequential(
            self.get_encoder(),
            TransformerClassificationHead(
                self.embed_size, n_classes, hidden_factor=hidden_factor),
        )

    def get_frozen_enc_model(self, n_classes, hidden_factor=2):
        enc = self.get_encoder()
        for param in enc.parameters():
            param.requires_grad = False

        return Sequential(
            enc,
            TransformerClassificationHead(
                self.embed_size, n_classes, hidden_factor=hidden_factor),
        )

    def get_finetune_model2(self, embedder, n_classes, hidden_factor=2):
        self.encoder.return_sides = False
        enc = deepcopy(self.encoder)
        return Sequential(
            embedder,
            enc,
            TransformerClassificationHead(
                self.embed_size, n_classes, hidden_factor=hidden_factor),
        )

    def get_frozen_enc_model2(self, embedder, n_classes, hidden_factor=2):
        self.encoder.return_sides = False
        enc = deepcopy(self.encoder)
        for param in enc.parameters():
            param.requires_grad = False

        return Sequential(
            embedder,
            enc,
            TransformerClassificationHead(
                self.embed_size, n_classes, hidden_factor=hidden_factor),
        )


class MultiOutFramework(BaseFrameWork):
    def __init__(self, params, modules, embed_size, n_subjects):
        super().__init__(params, modules, embed_size, n_subjects)

    def get_finetune_model(self, n_classes, hidden_factor=2):
        input_size = self.param('n_enc_tokens') * self.embed_size
        return Sequential(
            self.get_encoder(),
            FlattenHead(input_size, self.embed_size, n_classes)
        )

    def get_frozen_enc_model(self, n_classes, hidden_factor=2):
        enc = self.get_encoder()
        for param in enc.parameters():
            param.requires_grad = False

        input_size = self.param('n_enc_tokens') * self.embed_size
        return Sequential(
            self.get_encoder(),
            FlattenHead(input_size, self.embed_size, n_classes)
        )


############################################## FRAMEWORKS ##############################################

class D2VBERT(BaseFrameWork):
    def __init__(self, params, modules, embed_size, n_subjects):
        super().__init__(params, modules, embed_size, n_subjects)

        self.encoder.return_sides = True
        self.student = self.encoder
        self.teacher = GradualEmaModel(
            self.student, min_tau=self.param('min_tau'), max_tau=self.param('max_tau'), n_steps=self.param('tau_steps'))

        self.projection1 = self.head
        self.projection2 = deepcopy(self.head)

        self.criterion = SmoothL1Loss(self.param('beta'))

    def forward(self, x, label, subject):
        self.teacher.update(self.student)

        x = self.embedder(x)

        x_masked, mask = self.masker(x)

        enc_out, _ = self.student(x_masked)
        z1 = self.projection1(enc_out)

        x_hat = self.projection2(enc_out)

        z2 = self.get_teacher_targets(self.teacher, x, self.param('k'))

        loss = self.criterion(z1[mask], z2[mask]) + self.param('lambda') * \
            self.criterion(x_hat[mask], x[mask].detach().clone())
        return loss


class SimCLR2(MultiOutFramework):
    def __init__(self, params, modules, embed_size, n_subjects):
        super().__init__(params, modules, embed_size, n_subjects)

        self.encoder = MultiTokenOutEncoder(
            self.encoder, n_tokens=self.param('n_enc_tokens'))

        self.criterion = NT_Xent()

    def forward(self, x, label, subject):
        x = self.embedder(x)

        x_masked1, _ = self.masker(x)
        x_masked2, _ = self.masker(x)

        z1 = self.encoder(x_masked1)
        z1 = torch.flatten(z1, start_dim=-2)
        z1 = self.head(z1)

        z2 = self.encoder(x_masked2)
        z2 = torch.flatten(z2, start_dim=-2)
        z2 = self.head(z2)

        loss = self.criterion(z1, z2)
        return loss


class D2V2(MultiOutFramework):
    def __init__(self, params, modules, embed_size, n_subjects):
        super().__init__(params, modules, embed_size, n_subjects)

        self.encoder.return_sides = True
        self.encoder = MultiTokenOutEncoder(
            self.encoder, n_tokens=self.param('n_enc_tokens'))

        self.student = self.encoder
        self.teacher = GradualEmaModel(
            self.student, min_tau=self.param('min_tau'), max_tau=self.param('max_tau'), n_steps=self.param('tau_steps'))

        self.criterion = SmoothL1Loss(self.param('beta'))

    def forward(self, x, label, subject):
        self.teacher.update(self.student)

        x = self.embedder(x)

        x_masked, mask = self.masker(x)

        z1, _ = self.student(x_masked)
        z1 = self.head(z1)

        z2 = self.get_teacher_targets(self.teacher, x, self.param('k'))

        z1 = torch.flatten(z1, start_dim=-2)
        z2 = torch.flatten(z2, start_dim=-2)
        loss = self.criterion(z1, z2)
        return loss


class SubjectInputD2V(BaseFrameWork):
    def __init__(self, params, modules, embed_size, n_subjects):
        super().__init__(params, modules, embed_size, n_subjects)

        self.encoder.return_sides = True
        self.student = self.encoder
        self.teacher = GradualEmaModel(
            self.student, min_tau=self.param('min_tau'), max_tau=self.param('max_tau'), n_steps=self.param('tau_steps'))

        self.embedder = EmbeddingWithSubject(
            self.embedder, embed_size, n_subjects)
        self.criterion = SmoothL1Loss(self.param('beta'))

    def forward(self, x, label, subject):
        self.teacher.update(self.student)

        x = self.embedder(x, subject)

        x_masked, mask = self.masker(x)

        z1, _ = self.student(x_masked)
        z1 = self.head(z1)

        z2 = self.get_teacher_targets(self.teacher, x, self.param('k'))

        loss = self.criterion(z1[mask], z2[mask])
        return loss


class SubjectTokenD2V(BaseFrameWork):
    def __init__(self, params, modules, embed_size, n_subjects):
        super().__init__(params, modules, embed_size, n_subjects)

        self.encoder.return_sides = True
        self.student = self.encoder
        self.teacher = GradualEmaModel(
            self.student, min_tau=self.param('min_tau'), max_tau=self.param('max_tau'), n_steps=self.param('tau_steps'))

        self.subject_head = nn.Sequential(
            nn.Linear(self.embed_size, self.embed_size),
            nn.GELU(),
            nn.Linear(self.embed_size, self.n_subjects),
            nn.Softmax(dim=-1)
        )
        self.subject_token = nn.Parameter(torch.randn(1, 1, self.embed_size))

        self.criterion = SmoothL1Loss(self.param('beta'))
        self.subjec_criterion = nn.CrossEntropyLoss()

    def forward(self, x, label, subject):
        self.teacher.update(self.student)

        x = self.embedder(x)

        subject_token = repeat(
            self.subject_token, '() c e -> b c e', b=x.size(0))
        x = torch.cat([subject_token, x], dim=1)

        x_masked, mask = self.masker(x)

        z1, _ = self.student(x_masked)

        subject_out = z1[:, 0]
        subject_out = self.subject_head(subject_out)

        z1 = self.head(z1)

        z2 = self.get_teacher_targets(self.teacher, x, self.param('k'))

        loss = self.criterion(z1[mask], z2[mask]) + \
            self.param('lambda') * \
            self.subjec_criterion(subject_out, subject)
        return loss


class SubjectD2V(BaseFrameWork):
    def __init__(self, params, modules, embed_size, n_subjects):
        super().__init__(params, modules, embed_size, n_subjects)

        self.encoder.return_sides = True
        self.student = self.encoder
        self.teacher = GradualEmaModel(
            self.student, min_tau=self.param('min_tau'), max_tau=self.param('max_tau'), n_steps=self.param('tau_steps'))

        self.subject_head = nn.Sequential(
            nn.Linear(self.embed_size, self.embed_size),
            nn.GELU(),
            nn.Linear(self.embed_size, self.n_subjects),
            nn.Softmax(dim=-1)
        )

        self.criterion = SmoothL1Loss(self.param('beta'))
        self.subjec_criterion = nn.CrossEntropyLoss()

    def forward(self, x, label, subject):
        self.teacher.update(self.student)

        x = self.embedder(x)

        x_masked, mask = self.masker(x)

        z1, _ = self.student(x_masked)

        subject_out = torch.mean(z1, dim=-2)
        subject_out = self.subject_head(subject_out)

        z1 = self.head(z1)

        z2 = self.get_teacher_targets(self.teacher, x, self.param('k'))

        loss = self.criterion(z1[mask], z2[mask]) + \
            self.param('lambda') * \
            self.subjec_criterion(subject_out, subject)
        return loss


class D2V(BaseFrameWork):
    def __init__(self, params, modules, embed_size, n_subjects):
        super().__init__(params, modules, embed_size, n_subjects)

        self.encoder.return_sides = True
        self.student = self.encoder
        self.teacher = GradualEmaModel(
            self.student, min_tau=self.param('min_tau'), max_tau=self.param('max_tau'), n_steps=self.param('tau_steps'))

        self.criterion = SmoothL1Loss(self.param('beta'))

    def forward(self, x, label, subject):
        self.teacher.update(self.student)

        x = self.embedder(x)

        x_masked, mask = self.masker(x)

        z1, _ = self.student(x_masked)
        z1 = self.head(z1)

        z2 = self.get_teacher_targets(self.teacher, x, self.param('k'))

        loss = self.criterion(z1[mask], z2[mask])
        return loss


class BYOL(BaseFrameWork):
    def __init__(self, params, modules, embed_size, n_subjects):
        super().__init__(params, modules, embed_size, n_subjects)

        self.projection = deepcopy(self.head)

        self.student = Sequential(self.encoder, self.projection)
        self.teacher = GradualEmaModel(
            self.student, min_tau=self.param('min_tau'), max_tau=self.param('max_tau'), n_steps=self.param('tau_steps'))

        self.criterion = BYOLLoss()

    def forward(self, x, label, subject):
        self.teacher.update(self.student)

        x = self.embedder(x)

        x_masked1, _ = self.masker(x)
        x_masked2, _ = self.masker(x)

        z1 = self.student(x_masked1)
        z1 = self.head(z1)

        z2 = self.student(x_masked2)
        z2 = self.head(z2)

        with torch.no_grad():
            t1 = self.teacher(x_masked1).detach().clone()
            t2 = self.teacher(x_masked2).detach().clone()

        loss = self.criterion(z1, t2) + self.criterion(z2, t1)
        return loss


class SimCLR(BaseFrameWork):
    def __init__(self, params, modules, embed_size, n_subjects):
        super().__init__(params, modules, embed_size, n_subjects)

        self.criterion = NT_Xent()

    def forward(self, x, label, subject):
        x = self.embedder(x)

        x_masked1, _ = self.masker(x)
        x_masked2, _ = self.masker(x)

        z1 = self.encoder(x_masked1)
        z1 = self.head(z1)
        z1 = reduce(z1, 'b c e -> b e', reduction='mean')

        z2 = self.encoder(x_masked2)
        z2 = self.head(z2)
        z2 = reduce(z2, 'b c e -> b e', reduction='mean')

        loss = self.criterion(z1, z2)
        return loss


class SubjectBERT(BaseFrameWork):
    def __init__(self, params, modules, embed_size, n_subjects):
        super().__init__(params, modules, embed_size, n_subjects)

        self.embedder = EmbeddingWithSubject(
            self.embedder, embed_size, n_subjects)

        self.criterion = SmoothL1Loss(self.param('beta'))

    def forward(self, x, label, subject):
        x = self.embedder(x, subject)
        target = x.detach().clone()

        x_masked, mask = self.masker(x)

        z1 = self.encoder(x_masked)
        z1 = self.head(z1)

        loss = self.criterion(z1, target)
        return loss


class DenoisingAutoEncoder(BaseFrameWork):
    def __init__(self, params, modules, embed_size, n_subjects):
        super().__init__(params, modules, embed_size, n_subjects)

        self.criterion = SmoothL1Loss(self.param('beta'))
        self.to_token = EEGtoPatchTokens(self.param('patch_size'))

    def forward(self, x, label, subject):
        target = x.detach().clone()
        target = self.to_token(target)

        x = self.embedder(x)

        x_masked, mask = self.masker(x)

        z1 = self.encoder(x_masked)
        z1 = self.head(z1)

        loss = self.criterion(z1, target)
        return loss


class BERT(BaseFrameWork):
    def __init__(self, params, modules, embed_size, n_subjects):
        super().__init__(params, modules, embed_size, n_subjects)

        self.criterion = SmoothL1Loss(self.param('beta'))

    def forward(self, x, label, subject):
        x = self.embedder(x)
        target = x.detach().clone()

        x_masked, mask = self.masker(x)

        z1 = self.encoder(x_masked)
        z1 = self.head(z1)

        loss = self.criterion(z1, target)
        return loss


class SupervisedTrain(pl.LightningModule):
    def __init__(self, model, optimizer, scheduler=None, use_subject=False, n_classes=None):
        super().__init__()
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.model = model
        self.use_subject = use_subject

        self.criterion = nn.NLLLoss()
        self.train_acc = torchmetrics.Accuracy()
        self.valid_acc = torchmetrics.Accuracy()
        if n_classes:
            self.pres = torchmetrics.Precision(
                average='macro', num_classes=n_classes)
            self.rec = torchmetrics.Recall(
                average='macro', num_classes=n_classes)
            self.bac = torchmetrics.Accuracy(
                average='weighted', num_classes=n_classes)
        else:
            self.pres = torchmetrics.Precision(
                average='micro')
            self.rec = torchmetrics.Recall(
                average='micro')
            self.bac = torchmetrics.Accuracy(
                average='micro')

    def configure_optimizers(self):
        if self.scheduler is None:
            return self.optimizer

        scheduler_conf = {
            "scheduler": self.scheduler,
            "frequency": 1,
            "interval": "step",
        }
        return [self.optimizer], [scheduler_conf]

    def training_step(self, batch, batch_idx):
        x, label, subject = batch

        if self.use_subject:
            z = self.model(x, subject)
        else:
            z = self.model(x)
        loss = self.criterion(z, label)

        acc = self.train_acc(z, label)
        self.log('transfer_train_loss', loss)

        return loss

    def training_epoch_end(self, outputs):
        self.log('transfer_train_acc', self.train_acc.compute(), prog_bar=True)
        self.train_acc.reset()

    def validation_step(self, batch, batch_idx):
        x, label, subject = batch

        if self.use_subject:
            z = self.model(x, subject)
        else:
            z = self.model(x)

        loss = self.criterion(z, label)

        self.log('transfer_val_loss', loss)
        self.valid_acc.update(z, label)
        self.pres.update(z, label)
        self.rec.update(z, label)
        self.bac.update(z, label)

    def validation_epoch_end(self, outputs):
        self.log('transfer_val_acc', self.valid_acc.compute(), prog_bar=True)
        self.valid_acc.reset()

        self.log('precision', self.pres.compute(), prog_bar=True)
        self.pres.reset()

        self.log('recall', self.rec.compute(), prog_bar=True)
        self.rec.reset()

        self.log('balanced_acc', self.bac.compute(), prog_bar=True)
        self.bac.reset()


class PLWrapper(pl.LightningModule):
    def __init__(self, framework, optimizer, scheduler=None, save_interval=1000):
        super().__init__()

        self.save_interval = save_interval
        self.framework = framework
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.count = 0
        self.save_path = path = os.path.join(
            os.getcwd(), "models")
        os.makedirs(self.save_path, exist_ok=True)

    def configure_optimizers(self):
        if self.scheduler is None:
            return self.optimizer

        scheduler_conf = {
            "scheduler": self.scheduler,
            "frequency": 1,
            "interval": "step",
        }
        return [self.optimizer], [scheduler_conf]

    def training_step(self, train_batch, batch_idx):
        self.count += 1
        if self.count % self.save_interval == 0:
            save_file = os.path.join(self.save_path, f'temp_{self.count}.pt')
            torch.save(self.framework.encoder.state_dict(), save_file)
            wandb.save(save_file)

        x, label, subject = train_batch
        loss = self.framework(x, label, subject)
        self.log('train_loss', loss)
        return loss


def get_base_params(conf):
    min_tau = conf['min_tau']
    max_tau = conf['max_tau']
    k = conf['k']
    beta = conf['beta']
    tau_steps = conf['tau_steps']

    return min_tau, max_tau, k, beta, tau_steps


def get_framework(framework_info, embed_size, embedding, encoder, mask, head_info, data_exp, n_subjects):
    factories = {
        "subject-input-bert": SubjectBERT,
        "data2vec": D2V,
        "d2v2": D2V2,
        "d2vbert": D2VBERT,
        "denoise": DenoisingAutoEncoder,
        "simclr2": SimCLR2,
        "subjectd2v": SubjectD2V,
        "subject-token-d2v": SubjectTokenD2V,
        "subject-input-d2v": SubjectInputD2V,
        "byol": BYOL,
        "simclr": SimCLR,
        "bert": BERT
    }

    framework_name = framework_info["type"]
    params = framework_info["parameters"]

    # Maybe make abstract transformer class and
    # check in get_head and get_encoder wether
    # framework belongs to that w.r.t. return_sides?
    head = get_head(head_info, embed_size, data_exp)

    modules = {'embedding': embedding,
               'encoder': encoder,
               'head': head,
               'mask': mask
               }

    if framework_name in factories:
        return factories[framework_name](params, modules, embed_size, n_subjects)
    print(f"Unknown framework option: {framework_name}.")
