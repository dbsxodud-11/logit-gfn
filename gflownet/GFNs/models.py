import numpy as np
import torch
from torch_scatter import scatter_sum
import wandb

from .basegfn import BaseTBGFlowNet, tensor_to_np
from .condgfn import CondTBGFlowNet
from .. import network

class Empty(BaseTBGFlowNet):
  def __init__(self, args, mdp, actor):
    super().__init__(args, mdp, actor)
  
  def train(self, batch):
    return


class CondTBGFN(CondTBGFlowNet):
  """ Trajectory balance GFN. Learns forward and backward policy. """
  def __init__(self, args, mdp, actor):
    super().__init__(args, mdp, actor)
    print('Model: Cond_TBGFN')

  def train(self, batch, temp=None, online=False):
    return self.train_tb(batch, temp=temp, online=online)


class TBGFN(BaseTBGFlowNet):
  """ Trajectory balance GFN. Learns forward and backward policy. """
  def __init__(self, args, mdp, actor):
    super().__init__(args, mdp, actor)
    print('Model: TBGFN')

  def train(self, batch, online=False):
    return self.train_tb(batch, online=online)

  
class CondSubTBGFN(CondTBGFlowNet):
  """ Trajectory balance GFN. Learns forward and backward policy. """
  def __init__(self, args, mdp, actor):
    super().__init__(args, mdp, actor, cond=True)
    print('Model: Cond_SubTBGFN')
    
    if self.args.temp_cond_type == "layer" and self.args.thermometer:
      net = network.make_mlp(
        [actor.ft_dim + args.num_thermometer_dim] + \
        [args.sa_hid_dim] * args.sa_n_layers + \
        [1]
      )
    else:
      net = network.make_mlp(
        [actor.ft_dim + 1] + \
        [args.sa_hid_dim] * args.sa_n_layers + \
        [1]
      )
    self.logF = network.StateFeaturizeWrap(net, self.actor.featurize)
    self.logF.to(args.device)
    
    self.nets.append(self.logF)
    self.clip_grad_norm_params.append(list(self.logF.parameters()))
    
    self.optimizer_logF = torch.optim.Adam([
      {
        'params': self.logF.parameters(),
        'lr': args.lr_logF
      }])
    
    self.optimizers.append(self.optimizer_logF)
    print("define state flow estimation")
    
  def init_subtb(self):
    r"""Precompute all possible subtrajectory indices that we will use for computing the loss:
    \sum_{m=1}^{T-1} \sum_{n=m+1}^T
        \log( \frac{F(s_m) \prod_{i=m}^{n-1} P_F(s_{i+1}|s_i)}
                    {F(s_n) \prod_{i=m}^{n-1} P_B(s_i|s_{i+1})} )^2
    """
    self.subtb_max_len = self.mdp.forced_stop_len + 2
    ar = torch.arange(self.subtb_max_len, device=self.args.device)
    # This will contain a sequence of repeated ranges, e.g.
    # tidx[4] == tensor([0, 0, 1, 0, 1, 2, 0, 1, 2, 3])
    tidx = [torch.tril_indices(i, i, device=self.args.device)[1] for i in range(self.subtb_max_len)]
    # We need two sets of indices, the first are the source indices, the second the destination
    # indices. We precompute such indices for every possible trajectory length.

    # The source indices indicate where we index P_F and P_B, e.g. for m=3 and n=6 we'd need the
    # sequence [3,4,5]. We'll simply concatenate all sequences, for every m and n (because we're
    # computing \sum_{m=1}^{T-1} \sum_{n=m+1}^T), and get [0, 0,1, 0,1,2, ..., 3,4,5, ...].

    # The destination indices indicate the index of the subsequence the source indices correspond to.
    # This is used in the scatter sum to compute \log\prod_{i=m}^{n-1}. For the above example, we'd get
    # [0, 1,1, 2,2,2, ..., 17,17,17, ...]

    # And so with these indices, for example for m=0, n=3, the forward probability
    # of that subtrajectory gets computed as result[2] = P_F[0] + P_F[1] + P_F[2].

    self.precomp = [
        (
            torch.cat([i + tidx[T - i] for i in range(T)]),
            torch.cat(
                [ar[: T - i].repeat_interleave(ar[: T - i] + 1) + ar[T - i + 1 : T + 1].sum() for i in range(T)]
            ),
        )
        for T in range(1, self.subtb_max_len)
    ]
    self.lamda = self.args.lamda
    
  def train(self, batch, temp=None, online=False):
    self.init_subtb()
    return self.train_subtb(batch, temp=temp, online=online)
  
  def train_subtb(self, batch, log=True, temp=None, online=False):
    """ Step on trajectory balance loss.

      Parameters
      ----------
      batch: List of [Experience]

      Batching is handled in trainers.py.
    """
    batch_loss = self.batch_loss_detailed_balance(batch, temp=temp)

    for opt in self.optimizers:
      opt.zero_grad()
  
    batch_loss.backward()
    
    for param_set in self.clip_grad_norm_params:
      torch.nn.utils.clip_grad_norm_(param_set, self.args.clip_grad_norm)
    for opt in self.optimizers:
      opt.step()
    # self.clamp_logZ()

    if log:
      if online:
        self.online_loss_step += 1
        print(f'Online TB training:', batch_loss.item())
        wandb.log({'Online Regular TB loss': batch_loss.item(), 
                   'online_loss_step': self.online_loss_step})
      else:
        self.offline_loss_step += 1
        print(f'Offline TB training:', batch_loss.item())
        wandb.log({'Offline Regular TB loss': batch_loss.item(), 
                   'offline_loss_step': self.offline_loss_step})
    return
  
  def batch_loss_sub_trajectory_balance(self, batch, temp=None, temp_original=None):
    """ batch: List of [Experience].

        Calls fwd_logps_unique and back_logps_unique (gpu) in parallel on
        all states in all trajs in batch, then collates.
    """
    log_F_s, log_pf_actions = self.batch_traj_fwd_logp_unroll(batch, temp=temp, temp_original=temp_original)
    log_F_next_s, log_pb_actions = self.batch_traj_back_logp_unroll(batch, temp=temp, temp_original=temp_original)
    
    log_F_s[:, 0] = self.logZ.repeat(len(batch))
    for i, exp in enumerate(batch):
      log_F_next_s[i, -1] = temp_original[i] * exp.logr.clone().detach()
      
    total_loss = torch.zeros(len(batch), device=self.args.device)
    ar = torch.arange(self.subtb_max_len)
    for i in range(len(batch)):
      # Luckily, we have a fixed terminal length
      idces, dests = self.precomp[-1]
      P_F_sums = scatter_sum(log_pf_actions[i, idces], dests)
      P_B_sums = scatter_sum(log_pb_actions[i, idces], dests)
      F_start = scatter_sum(log_F_s[i, idces], dests)
      F_end = scatter_sum(log_F_next_s[i, idces], dests)

      weight = torch.pow(self.lamda, torch.bincount(dests) - 1)
      total_loss[i] = (weight * (F_start - F_end + P_F_sums - P_B_sums).pow(2)).sum() / torch.sum(weight)
    # losses = torch.clamp(total_loss, max=5000)
    mean_loss = torch.mean(total_loss)
    return mean_loss


class SubTBGFN(BaseTBGFlowNet):
  """ SubTB (lambda) """
  def __init__(self, args, mdp, actor):
    super().__init__(args, mdp, actor)
    print('Model: SubTBGFN')
    
    net = network.make_mlp(
      [actor.ft_dim] + \
      [args.sa_hid_dim] * args.sa_n_layers + \
      [1]
    )
    self.logF = network.StateFeaturizeWrap(net, self.actor.featurize)
    self.logF.to(args.device)
    
    self.nets.append(self.logF)
    self.clip_grad_norm_params.append(self.logF.parameters())
    
    self.optimizer_logF = torch.optim.Adam([
      {
        'params': self.logF.parameters(),
        'lr': args.lr_logF
      }])
    
    self.optimizers.append(self.optimizer_logF)
    print("define state flow estimation")
    pass
    
  def init_subtb(self):
    r"""Precompute all possible subtrajectory indices that we will use for computing the loss:
    \sum_{m=1}^{T-1} \sum_{n=m+1}^T
        \log( \frac{F(s_m) \prod_{i=m}^{n-1} P_F(s_{i+1}|s_i)}
                    {F(s_n) \prod_{i=m}^{n-1} P_B(s_i|s_{i+1})} )^2
    """
    self.subtb_max_len = self.mdp.forced_stop_len + 2
    ar = torch.arange(self.subtb_max_len, device=self.args.device)
    # This will contain a sequence of repeated ranges, e.g.
    # tidx[4] == tensor([0, 0, 1, 0, 1, 2, 0, 1, 2, 3])
    tidx = [torch.tril_indices(i, i, device=self.args.device)[1] for i in range(self.subtb_max_len)]
    # We need two sets of indices, the first are the source indices, the second the destination
    # indices. We precompute such indices for every possible trajectory length.

    # The source indices indicate where we index P_F and P_B, e.g. for m=3 and n=6 we'd need the
    # sequence [3,4,5]. We'll simply concatenate all sequences, for every m and n (because we're
    # computing \sum_{m=1}^{T-1} \sum_{n=m+1}^T), and get [0, 0,1, 0,1,2, ..., 3,4,5, ...].

    # The destination indices indicate the index of the subsequence the source indices correspond to.
    # This is used in the scatter sum to compute \log\prod_{i=m}^{n-1}. For the above example, we'd get
    # [0, 1,1, 2,2,2, ..., 17,17,17, ...]

    # And so with these indices, for example for m=0, n=3, the forward probability
    # of that subtrajectory gets computed as result[2] = P_F[0] + P_F[1] + P_F[2].

    self.precomp = [
        (
            torch.cat([i + tidx[T - i] for i in range(T)]),
            torch.cat(
                [ar[: T - i].repeat_interleave(ar[: T - i] + 1) + ar[T - i + 1 : T + 1].sum() for i in range(T)]
            ),
        )
        for T in range(1, self.subtb_max_len)
    ]
    self.lamda = self.args.lamda
    
  def train(self, batch, online=False):
    self.init_subtb()
    return self.train_subtb(batch, online=online)
  
  def train_subtb(self, batch, log = True, online=False):
    """ Step on subtrajectory balance loss.

      Parameters
      ----------
      batch: List of [Experience]

      Batching is handled in trainers.py.
    """
    batch_loss = self.batch_loss_sub_trajectory_balance(batch)

    for opt in self.optimizers:
      opt.zero_grad()
  
    batch_loss.backward()
    
    for param_set in self.clip_grad_norm_params:
      # torch.nn.utils.clip_grad_norm_(param_set, self.args.clip_grad_norm, error_if_nonfinite=True)
      torch.nn.utils.clip_grad_norm_(param_set, self.args.clip_grad_norm)
    for opt in self.optimizers:
      opt.step()
    self.clamp_logZ()

    if log:
      if online:
        self.online_loss_step += 1
        print(f'Online TB training:', batch_loss.item())
        wandb.log({'Online Regular TB loss': batch_loss.item(),
                   'online_loss_step': self.online_loss_step})
      else:
        self.offline_loss_step += 1
        print(f'Offline TB training:', batch_loss.item())
        wandb.log({'Offline Regular TB loss': batch_loss.item(),
                   'offline_loss_step': self.offline_loss_step})
    return
  
  def batch_loss_sub_trajectory_balance(self, batch):
    """ batch: List of [Experience].

        Calls fwd_logps_unique and back_logps_unique (gpu) in parallel on
        all states in all trajs in batch, then collates.
    """
    log_F_s, log_pf_actions = self.batch_traj_fwd_logp_unroll(batch)
    log_F_next_s, log_pb_actions = self.batch_traj_back_logp_unroll(batch)
    
    log_F_s[:, 0] = self.logZ.repeat(len(batch))
    for i, exp in enumerate(batch):
      log_F_next_s[i, -1] = self.args.target_beta * exp.logr.clone().detach()
      
    total_loss = torch.zeros(len(batch), device=self.args.device)
    ar = torch.arange(self.subtb_max_len)
    for i in range(len(batch)):
      # Luckily, we have a fixed terminal length
      idces, dests = self.precomp[-1]
      P_F_sums = scatter_sum(log_pf_actions[i, idces], dests)
      P_B_sums = scatter_sum(log_pb_actions[i, idces], dests)
      F_start = scatter_sum(log_F_s[i, idces], dests)
      F_end = scatter_sum(log_F_next_s[i, idces], dests)

      weight = torch.pow(self.lamda, torch.bincount(dests) - 1)
      total_loss[i] = (weight * (F_start - F_end + P_F_sums - P_B_sums).pow(2)).sum() / torch.sum(weight)
    # losses = torch.clamp(total_loss, max=5000)
    mean_loss = torch.mean(total_loss)
    return mean_loss
  

class CondDBGFN(CondTBGFlowNet):
  """ Detailed balance GFN """
  def __init__(self, args, mdp, actor):
    super().__init__(args, mdp, actor)
    print('Model: Cond_DBGFN') 
    
    if self.args.temp_cond_type == "layer" and self.args.thermometer:
      net = network.make_mlp(
        [actor.ft_dim + args.num_thermometer_dim] + \
        [args.sa_hid_dim] * args.sa_n_layers + \
        [1]
      )
    else:
      net = network.make_mlp(
        [actor.ft_dim + 1] + \
        [args.sa_hid_dim] * args.sa_n_layers + \
        [1]
      )
    self.logF = network.StateFeaturizeWrap(net, self.actor.featurize)
    self.logF.to(args.device)
    
    self.nets.append(self.logF)
    self.clip_grad_norm_params.append(list(self.logF.parameters()))
    
    self.optimizer_logF = torch.optim.Adam([
      {
        'params': self.logF.parameters(),
        'lr': args.lr_logF
      }])
    
    self.optimizers.append(self.optimizer_logF)
    print("define state flow estimation")
    
  def train(self, batch, temp=None, online=False):
    return self.train_db(batch, temp=temp, online=online)
  
  def train_db(self, batch, log=True, temp=None, online=False):
    """ Step on detailed balance loss.

      Parameters
      ----------
      batch: List of [Experience]

      Batching is handled in trainers.py.
    """
    batch_loss = self.batch_loss_detailed_balance(batch, temp=temp)

    for opt in self.optimizers:
      opt.zero_grad()
  
    batch_loss.backward()
    
    for param_set in self.clip_grad_norm_params:
      torch.nn.utils.clip_grad_norm_(param_set, self.args.clip_grad_norm)
    for opt in self.optimizers:
      opt.step()
    # self.clamp_logZ()

    if log:
      if online:
        self.online_loss_step += 1
        print(f'Online TB training:', batch_loss.item())
        wandb.log({'Online Regular TB loss': batch_loss.item(), 
                   'online_loss_step': self.online_loss_step})
      else:
        self.offline_loss_step += 1
        print(f'Offline TB training:', batch_loss.item())
        wandb.log({'Offline Regular TB loss': batch_loss.item(), 
                   'offline_loss_step': self.offline_loss_step})
    return
  
  def batch_loss_detailed_balance(self, batch, temp=None):
    """ batch: List of [Experience].

        Calls fwd_logps_unique and back_logps_unique (gpu) in parallel on
        all states in all trajs in batch, then collates.
    """
    log_F_s, log_pf_actions = self.batch_traj_fwd_logp_unroll(batch, temp)
    log_F_next_s, log_pb_actions = self.batch_traj_back_logp_unroll(batch, temp)
    
    for i, exp in enumerate(batch):
      log_F_next_s[i, -1] = temp["beta"].squeeze(-1)[i] * exp.logr.clone().detach()

    losses = (log_F_s + log_pf_actions - log_F_next_s - log_pb_actions).pow(2).sum(axis=1)
    # losses = torch.clamp(losses, max=5000)
    mean_loss = torch.mean(losses)
    return mean_loss


class DBGFN(BaseTBGFlowNet):
  """ Detailed balance GFN """
  def __init__(self, args, mdp, actor):
    super().__init__(args, mdp, actor)
    print('Model: DBGFN')
    
    net = network.make_mlp(
      [actor.ft_dim] + \
      [args.sa_hid_dim] * args.sa_n_layers + \
      [1]
    )
    self.logF = network.StateFeaturizeWrap(net, self.actor.featurize)
    self.logF.to(args.device)
    
    self.nets.append(self.logF)
    self.clip_grad_norm_params.append(list(self.logF.parameters()))
    
    self.optimizer_logF = torch.optim.Adam([
      {
        'params': self.logF.parameters(),
        'lr': args.lr_logF
      }])
    
    self.optimizers.append(self.optimizer_logF)
    print("define state flow estimation")
    pass
    
  def train(self, batch, temp=None, online=False):
    return self.train_db(batch, temp, online)
  
  def train_db(self, batch, log = True, online=False):
    """ Step on detailed balance loss.

      Parameters
      ----------
      batch: List of [Experience]

      Batching is handled in trainers.py.
    """
    batch_loss = self.batch_loss_detailed_balance(batch)

    for opt in self.optimizers:
      opt.zero_grad()
  
    batch_loss.backward()
    
    for param_set in self.clip_grad_norm_params:
      # torch.nn.utils.clip_grad_norm_(param_set, self.args.clip_grad_norm, error_if_nonfinite=True)
      torch.nn.utils.clip_grad_norm_(param_set, self.args.clip_grad_norm)
    for opt in self.optimizers:
      opt.step()
    self.clamp_logZ()

    if log:
      if online:
        self.online_loss_step += 1
        print(f'Online TB training:', batch_loss.item())
        wandb.log({'Online Regular TB loss': batch_loss.item(),
                   'online_loss_step': self.online_loss_step})
      else:
        self.offline_loss_step += 1
        print(f'Offline TB training:', batch_loss.item())
        wandb.log({'Offline Regular TB loss': batch_loss.item(),
                   'offline_loss_step': self.offline_loss_step})
    return
  
  def batch_loss_detailed_balance(self, batch):
    """ batch: List of [Experience].

        Calls fwd_logps_unique and back_logps_unique (gpu) in parallel on
        all states in all trajs in batch, then collates.
    """
    log_F_s, log_pf_actions = self.batch_traj_fwd_logp_unroll(batch)
    log_F_next_s, log_pb_actions = self.batch_traj_back_logp_unroll(batch)
    
    for i, exp in enumerate(batch):
      log_F_next_s[i, -1] = self.args.target_beta * exp.logr.clone().detach()

    losses = (log_F_s + log_pf_actions - log_F_next_s - log_pb_actions).pow(2).sum(axis=1)
    # losses = torch.clamp(losses, max=5000)
    mean_loss = torch.mean(losses)
    return mean_loss


def make_model(args, mdp, actor):
  """ Constructs TB / DB / SubTB GFN. """
  if args.temp_cond:
    if args.loss_type == "tb":
      return CondTBGFN(args, mdp, actor)
    elif args.loss_type == "db":
      return CondDBGFN(args, mdp, actor)
    elif args.loss_type == "subtb":
      return CondSubTBGFN(args, mdp, actor)
  else:
    if args.loss_type == "tb":
      return TBGFN(args, mdp, actor)
    elif args.loss_type == "db":
      return DBGFN(args, mdp, actor)
    elif args.loss_type == "subtb":
      return SubTBGFN(args, mdp, actor)
