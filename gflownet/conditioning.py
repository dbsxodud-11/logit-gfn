import abc

import numpy as np
import torch
from torch import Tensor


def thermometer(v: Tensor, n_bins: int = 50, vmin: float = 0, vmax: float = 1) -> Tensor:
    """Thermometer encoding of a scalar quantity.

    Parameters
    ----------
    v: Tensor
        Value(s) to encode. Can be any shape
    n_bins: int
        The number of dimensions to encode the values into
    vmin: float
        The smallest value, below which the encoding is equal to torch.zeros(n_bins)
    vmax: float
        The largest value, beyond which the encoding is equal to torch.ones(n_bins)
    Returns
    -------
    encoding: Tensor
        The encoded values, shape: `v.shape + (n_bins,)`
    """
    bins = torch.linspace(vmin, vmax, n_bins)
    gap = bins[1] - bins[0]
    assert gap > 0, "vmin and vmax must be different"
    # return (v[..., None] - bins.reshape((1,) * v.ndim + (-1,))).clamp(0, gap.item()) / gap
    return (v[..., None] - bins.reshape((1,) * v.ndim + (-1,))) / gap


class Conditional(abc.ABC):
    def sample(self, n):
        raise NotImplementedError()


class TemperatureConditional(Conditional):
    def __init__(self, args):
        self.args = args
        self.rng = np.random.default_rng(142857)
        
        self.train_temp_dist = self.args.train_temp_dist
        if self.train_temp_dist == "constant":
            self.train_temp = self.args.train_temp
            self.train_temp_min = self.args.train_temp
            self.train_temp_max = self.args.train_temp
        elif self.train_temp_dist == "normal":
            self.train_temp_mu = self.args.train_temp_mu
            self.train_temp_sigma = self.args.train_temp_sigma
            
            self.train_temp_min = self.args.train_temp_min
            self.train_temp_max = self.args.train_temp_max
            assert self.train_temp_min <= self.train_temp_max
            assert self.train_temp_min <= self.train_temp_mu and self.train_temp_mu <= self.train_temp_max
        elif self.train_temp_dist.endswith("uniform"):
            self.train_temp_min = self.args.train_temp_min
            self.train_temp_max = self.args.train_temp_max
            assert self.train_temp_min <= self.train_temp_max
        elif self.train_temp_dist.startswith("annealing"):
            self.train_temp_min = self.args.train_temp_min
            self.train_temp_max = self.args.train_temp_max
            self.interval = 0.5
            assert self.train_temp_min <= self.train_temp_max
        else:
            raise NotImplementedError

        self.exp_temp_dist = self.args.exp_temp_dist
        if self.exp_temp_dist == "constant":
            self.exp_temp = self.args.exp_temp
            self.exp_temp_min = self.args.exp_temp
            self.exp_temp_max = self.args.exp_temp
        elif self.exp_temp_dist == "normal":
            self.exp_temp_mu = self.args.exp_temp_mu
            self.exp_temp_sigma = self.args.exp_temp_sigma
            
            self.exp_temp_min = self.args.exp_temp_min
            self.exp_temp_max = self.args.exp_temp_max
            assert self.exp_temp_min <= self.exp_temp_max
            assert self.exp_temp_min <= self.exp_temp_mu and self.exp_temp_mu <= self.exp_temp_max
        elif self.exp_temp_dist.endswith("uniform"):
            self.exp_temp_min = self.args.exp_temp_min
            self.exp_temp_max = self.args.exp_temp_max
            assert self.exp_temp_min <= self.exp_temp_max
        elif self.exp_temp_dist.startswith("annealing"):
            self.exp_temp_min = self.args.exp_temp_min
            self.exp_temp_max = self.args.exp_temp_max
            self.interval = 0.5
            assert self.exp_temp_min <= self.exp_temp_max
        else:
            raise NotImplementedError
        
    def sample(self, n, exp=False, fraction=None):
        if exp:
            if self.args.exp_temp_dist == "constant":
                beta = np.array(self.exp_temp).repeat(n).astype(np.float32)
            
            # static distributions
            elif self.args.exp_temp_dist == "normal":
                beta = self.rng.normal(self.exp_temp_mu, self.exp_temp_sigma, n).astype(np.float32)
                beta = np.clip(beta, self.exp_temp_min, self.exp_temp_max)
            elif self.args.exp_temp_dist == "uniform":
                beta = self.rng.uniform(self.exp_temp_min, self.exp_temp_max, n).astype(np.float32)
            elif self.args.exp_temp_dist == "loguniform":
                beta = np.log(self.rng.uniform(np.exp(self.exp_temp_min), np.exp(self.exp_temp_max), n)).astype(np.float32)
            elif self.args.exp_temp_dist == "expuniform":
                beta = np.exp(self.rng.uniform(np.log(self.exp_temp_min), np.log(self.exp_temp_max), n)).astype(np.float32)
            
            # dynamic distributions
            elif self.args.exp_temp_dist == "annealing":
                mu = self.exp_temp_min + (self.exp_temp_max - self.exp_temp_min) * fraction
                # mu = self.exp_temp_min + (self.exp_temp_max - self.exp_temp_min) * (np.exp(fraction) - 1) / (math.e - 1)
                beta = self.rng.uniform(mu - self.interval, mu + self.interval, n).astype(np.float32)
                beta = np.clip(beta, self.exp_temp_min, self.exp_temp_max)
            elif self.args.exp_temp_dist == "annealing-inv":
                mu = self.exp_temp_min + (self.exp_temp_max - self.exp_temp_min) * (1.0 - fraction)
                # mu = self.exp_temp_min + (self.exp_temp_max - self.exp_temp_min) * (np.exp((1.0 - fraction)) - 1) / (math.e - 1)
                beta = self.rng.uniform(mu - self.interval, mu + self.interval, n).astype(np.float32)
                beta = np.clip(beta, self.exp_temp_min, self.exp_temp_max)
            else:
                raise NotImplementedError
        else:
            if self.args.train_temp_dist == "constant":
                beta = np.array(self.train_temp).repeat(n).astype(np.float32)
            
            # static distributions
            elif self.args.train_temp_dist == "normal":
                beta = self.rng.normal(self.train_temp_mu, self.train_temp_sigma, n).astype(np.float32)
                beta = np.clip(beta, self.train_temp_min, self.train_temp_max)
            elif self.args.train_temp_dist == "uniform":
                beta = self.rng.uniform(self.train_temp_min, self.train_temp_max, n).astype(np.float32)
            elif self.args.train_temp_dist == "loguniform":
                beta = np.log(self.rng.uniform(np.exp(self.train_temp_min), np.exp(self.train_temp_max), n)).astype(np.float32)
            elif self.args.train_temp_dist == "expuniform":
                beta = np.exp(self.rng.uniform(np.log(self.train_temp_min), np.log(self.train_temp_max), n)).astype(np.float32)
            
            # dynamic distributions
            elif self.args.train_temp_dist == "annealing":
                mu = self.train_temp_min + (self.train_temp_max - self.train_temp_min) * fraction
                # mu = self.train_temp_min + (self.train_temp_max - self.train_temp_min) * (np.exp(fraction) - 1) / (math.e - 1)
                beta = self.rng.uniform(mu - self.interval, mu + self.interval, n).astype(np.float32)
                beta = np.clip(beta, self.train_temp_min, self.train_temp_max)
            elif self.args.train_temp_dist == "annealing-inv":
                mu = self.train_temp_min + (self.train_temp_max - self.train_temp_min) * (1.0 - fraction)
                # mu = self.train_temp_min + (self.train_temp_max - self.train_temp_min) * (np.exp((1.0 - fraction)) - 1) / (math.e - 1)
                beta = self.rng.uniform(mu - self.interval, mu + self.interval, n).astype(np.float32)
                beta = np.clip(beta, self.train_temp_min, self.train_temp_max)
            else:
                raise NotImplementedError            
        
        beta = torch.tensor(beta)
        if self.args.temp_cond_type == "layer" and self.args.thermometer:
            beta_enc = thermometer(beta, n_bins=self.args.num_thermometer_dim,
                                     vmin=min(self.train_temp_min, self.exp_temp_min),
                                     vmax=max(self.train_temp_max, self.exp_temp_max)).to(self.args.device)
            beta = beta.unsqueeze(1).to(self.args.device)
            return {"beta": beta, "beta_enc": beta_enc}
        else:
            beta = beta.unsqueeze(1).to(self.args.device)
            return {"beta": beta, "beta_enc": beta}
