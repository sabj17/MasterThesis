from torch.optim.optimizer import Optimizer, required
import torch
import torch.nn as nn
from copy import deepcopy


class GradualEmaModel(nn.Module):
    def __init__(self, model, min_tau=0.1, max_tau=0.99, n_steps=1e4):
        super().__init__()
        self.model_copy = deepcopy(model)

        self.tau = min_tau
        self.max_tau = max_tau

        self.change_rate = (max_tau - min_tau) / n_steps

    def update(self, model):
        with torch.no_grad():
            for ema_v, model_v in zip(self.model_copy.state_dict().values(), model.state_dict().values()):
                ema_v.copy_(self.tau * ema_v + (1. - self.tau) * model_v)

        if self.tau < self.max_tau:
            self.tau += self.change_rate

    def forward(self, x, *args):
        return self.model_copy(x, *args)


class EmaModel(nn.Module):
    def __init__(self, model, tau=0.999):
        super(EmaModel, self).__init__()
        self.model_copy = deepcopy(model)
        self.model_copy
        self.tau = tau

    def update(self, model):
        with torch.no_grad():
            for ema_v, model_v in zip(self.model_copy.state_dict().values(), model.state_dict().values()):
                ema_v.copy_(self.tau * ema_v + (1. - self.tau) * model_v)

    def forward(self, x, *args):
        return self.model_copy(x, *args)


class Sequential(nn.Sequential):
    def forward(self, *inputs):
        for module in self._modules.values():
            if type(inputs) == tuple:
                inputs = module(*inputs)
            else:
                inputs = module(inputs)
        return inputs


def warmup_schedule(optimizer, warmup_steps):
    def warm_decay(step):
        if step < warmup_steps:
            return step / warmup_steps
        return warmup_steps ** 0.5 * step ** -0.5

    return optim.lr_scheduler.LambdaLR(optimizer, warm_decay)


class NoamOpt:
    "Optim wrapper that implements rate."

    def __init__(self, model_size, factor, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0

    def step(self):
        "Update parameters and rate"
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate

    def rate(self, step=None):
        "Implement `lrate` above"
        if step is None:
            step = self._step
        return self.factor * \
            (self.model_size ** (-0.5) *
             min(step ** (-0.5), step * self.warmup ** (-1.5)))


class LARS(Optimizer):
    r"""Implements LARS (Layer-wise Adaptive Rate Scaling).

    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float): learning rate
        momentum (float, optional): momentum factor (default: 0)
        eta (float, optional): LARS coefficient as used in the paper (default: 1e-3)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        dampening (float, optional): dampening for momentum (default: 0)
        nesterov (bool, optional): enables Nesterov momentum (default: False)
        epsilon (float, optional): epsilon to prevent zero division (default: 0)

    Example:
        >>> optimizer = torch.optim.LARS(model.parameters(), lr=0.1, momentum=0.9)
    """

    def __init__(self, params, lr=required, momentum=0, eta=1e-3, dampening=0,
                 weight_decay=0, nesterov=False, epsilon=0):
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError(
                "Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(lr=lr, momentum=momentum, eta=eta, dampening=dampening,
                        weight_decay=weight_decay, nesterov=nesterov, epsilon=epsilon)
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError(
                "Nesterov momentum requires a momentum and zero dampening")
        super(LARS, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(LARS, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            eta = group['eta']
            dampening = group['dampening']
            nesterov = group['nesterov']
            epsilon = group['epsilon']

            for p in group['params']:
                if p.grad is None:
                    continue
                w_norm = torch.norm(p.data)
                g_norm = torch.norm(p.grad.data)
                if w_norm * g_norm > 0:
                    local_lr = eta * w_norm / (g_norm +
                                               weight_decay * w_norm + epsilon)
                else:
                    local_lr = 1
                d_p = p.grad.data
                if weight_decay != 0:
                    d_p.add_(weight_decay, p.data)
                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.clone(
                            d_p).detach()
                    else:
                        buf = param_state['momentum_buffer']
                    buf.mul_(momentum).add_(1 - dampening, d_p)
                    if nesterov:
                        d_p = d_p.add(momentum, buf)
                    else:
                        d_p = buf

                p.data.add_(-local_lr * group['lr'], d_p)

        return loss
