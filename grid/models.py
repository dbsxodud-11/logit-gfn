"""
Model for the GridWorld environment
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union, cast
import itertools

import numpy as np

import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical

from grid.env import GridEnv


#################
#    Modules    #
#################


class MLP(nn.Sequential):
    def __init__(
        self,
        dim_list: List[int],
        activation: nn.Module = nn.LeakyReLU(),
        tail: List[nn.Module] = [],
    ) -> None:
        super().__init__(
            *(
                list(
                    itertools.chain(
                        *[
                            [nn.Linear(i, o)] + ([activation] if n < len(dim_list) - 2 else [])
                            for n, (i, o) in enumerate(zip(dim_list, dim_list[1:]))
                        ]
                    )
                )
                + tail
            )
        )


#################
#    Buffers    #
#################


class ReplayBuffer:
    def __init__(
        self,
        env: GridEnv,
        buffer_size: int,
        sample_size: int,
        add_strategy: str = "top_k",
        sample_strategy: str = "random",
    ):
        self.env = env
        self.buffer_size = buffer_size
        self.sample_size = sample_size
        self.add_strategy = add_strategy
        self.sample_strategy = sample_strategy

        # Buffer
        self.buffer: List[Tuple[Tuple[int], float]] = []

    def add(self, x: Tuple[int], r_x: float) -> None:
        # x is a single terminal state
        if self.add_strategy == "top_k":
            if len(self.buffer) < self.buffer_size or r_x > self.buffer[0][1]:
                self.buffer = sorted(self.buffer + [(x, r_x)], key=lambda t: t[1])[-self.buffer_size :]
        else:
            raise NotImplementedError

    def sample(self) -> List[List[Tuple[torch.Tensor, ...]]]:
        if len(self.buffer) == 0:
            return []

        if self.sample_strategy == "random":
            idxs = np.random.randint(0, len(self.buffer), self.sample_size)
        else:
            raise NotImplementedError

        return list(itertools.chain(*[self.generate_backward(*self.buffer[i]) for i in idxs]))

    def generate_backward(self, x: Tuple[int], r_x: float):
        _x = np.int8(x)
        o_x = self.env.obs(_x)
        # If s0 is a forced-terminal state, the the action that leads
        # to it is s0.argmax() which .parents finds, but if it isn't,
        # we must indicate that the agent ended the trajectory with
        # the stop action
        used_stop_action = _x.max() < self.env.horizon - 1
        done = True
        # Now we work backward from that last transition
        traj = []
        while _x.sum() > 0:
            parents, actions = self.env.parent_transitions(_x, used_stop_action)
            # add the transition
            traj.append([torch.FloatTensor(i) for i in (parents, actions, [r_x], [self.env.obs(_x)], [done])])
            # Then randomly choose a parent state
            if not used_stop_action:
                i = np.random.randint(0, len(parents))
                a = actions[i]
                _x[a] -= 1
            # Values for intermediary trajectory states:
            used_stop_action = False
            done = False
            r_x = 0
        return traj


################
#    Agents    #
################


class BaseAgent(nn.Module, ABC):
    def __init__(
        self,
        envs: List[GridEnv],
        horizon: int,
        ndim: int,
        eps: float = 0.1,
        logit_temp: float = 1.0,   
        device: Union[torch.device, str] = "cpu",
        use_buffer: bool = False,
        buffer_params: Optional[dict] = None,
    ) -> None:
        super().__init__()
        self.envs = envs  # TODO: do not use list of envs but a single env
        self.horizon = horizon
        self.ndim = ndim
        self.eps = eps
        self.logit_temp = logit_temp
        self.device = device
        self.use_buffer = use_buffer
        self.buffer_params = buffer_params

        # Replay buffer
        self.use_buffer = use_buffer
        self.replay = None
        if use_buffer and buffer_params is not None:
            self.replay = ReplayBuffer(env=self.envs[0], **buffer_params)

        self.to_ft = lambda x: torch.tensor(x, dtype=torch.float, device=device)
        self.to_lt = lambda x: torch.tensor(x, dtype=torch.long, device=device)

    @abstractmethod
    def sample_many(self, batch_size: int, eval: bool = False, beta=1.0) -> Tuple[List[Tuple[torch.Tensor, ...]], List[Tuple]]:
        raise NotImplementedError

    @abstractmethod
    def learn_from(self, train_batch: List[Tuple[torch.Tensor, ...]]) -> Tuple[torch.Tensor, Dict[str, Any]]:
        raise NotImplementedError


class TBAgent(BaseAgent):
    """
    Agent for the GFlowNet with Trajectory-Balance (TB) loss
    reference for TB: https://arxiv.org/abs/2201.13259

    Three parameterizations:
        1. Total-flow, Z
        2. Forward transition probability, P_F(s' | s)
        3. Backward transition probability, P_B(s | s') ... this can be fixed
    """

    def __init__(
        self,
        envs: List[GridEnv],
        horizon: int,
        ndim: int,
        eps: float = 0.1,
        logit_temp: float = 1.0,
        device: Union[torch.device, str] = "cpu",
        use_buffer: bool = False,
        buffer_params: Optional[dict] = None,
        n_hidden: int = 256,
        n_layers: int = 2,
        tb_beta: float = 1.0,
    ) -> None:
        super().__init__(envs, horizon, ndim, eps, logit_temp, device, use_buffer, buffer_params)
        
        out_dim = 2 * ndim + 1  # x 2 for forward/backward, + 1 for terminating action
        self.model = MLP(dim_list=[horizon * ndim] + [n_hidden] * n_layers + [out_dim])
        self.model.to(device)

        self.log_Z = nn.parameter.Parameter(torch.zeros((), device=device, requires_grad=True))
        self.beta = tb_beta

    def sample_many(self, batch_size: int, eval: bool = False, beta=1.0) -> Tuple[List[Tuple[torch.Tensor, ...]], List[Tuple]]:
        # replayed = self.replay.sample() if self.use_buffer and self.replay else []
        # TODO: Use the replay buffer

        assert batch_size <= len(self.envs)

        # Save trajectory
        batch_traj_obs, batch_traj_s, batch_traj_a = (
            [[] for _ in range(batch_size)],
            [[] for _ in range(batch_size)],
            [[] for _ in range(batch_size)],
        )

        # Save step results
        batch_obs, batch_r, batch_done, batch_s = [list(x) for x in zip(*[e.reset() for e in self.envs[:batch_size]])]
        batch_obs = self.to_ft(batch_obs)
        batch_s = self.to_lt(batch_s)

        visited_x = []
        not_done_indices = [i for i in range(batch_size)]
        while not all(batch_done):
            with torch.no_grad():
                pred = cast(torch.Tensor, self.model(batch_obs))
                logit_temp = 1.0 if eval else self.logit_temp
                f_probs = (pred[..., : self.ndim + 1] / logit_temp).softmax(1)
                eps = 0.0 if eval else self.eps
                f_probs = (1 - eps) * f_probs + (eps) / (self.ndim + 1)
                actions = f_probs.multinomial(1)

            batch_step = [self.envs[i].step(int(a)) for i, a in zip(not_done_indices, actions)]

            for i, (curr_obs, curr_s, curr_a) in enumerate(zip(batch_obs, batch_s, actions)):
                env_idx = not_done_indices[i]
                batch_traj_obs[env_idx].append(curr_obs)
                batch_traj_s[env_idx].append(curr_s)
                batch_traj_a[env_idx].append(curr_a)

            new_batch_obs, new_batch_s, new_not_done_indices = [], [], []
            for i, (_obs, _r, _d, _s) in enumerate(batch_step):
                env_idx = not_done_indices[i]
                batch_done[env_idx] = _d

                if _d:
                    batch_r[env_idx] = _r
                    batch_traj_obs[env_idx].append(self.to_ft(_obs))
                    batch_traj_s[env_idx].append(self.to_lt(_s))
                    visited_x.append(tuple(_s))
                else:
                    new_batch_obs.append(_obs)
                    new_batch_s.append(_s)
                    new_not_done_indices.append(env_idx)

            batch_obs, batch_s = self.to_ft(new_batch_obs), self.to_lt(new_batch_s)
            not_done_indices = new_not_done_indices

        batch_traj_obs_t = list(map(torch.stack, batch_traj_obs))
        batch_traj_s_t = list(map(torch.stack, batch_traj_s))
        batch_traj_a_t = list(map(torch.stack, batch_traj_a))
        batch_r = list(map(self.to_ft, batch_r))

        train_batch = list(zip(batch_traj_obs_t, batch_traj_s_t, batch_traj_a_t, batch_r))

        return train_batch, visited_x

    def learn_from(self, train_batch: List[Tuple[torch.Tensor, ...]]) -> Tuple[torch.Tensor, Dict]:
        inf = 1000000000

        loss = torch.tensor(0.0, dtype=torch.float, device=self.device)
        for _obs, _s, _a, _r in train_batch:
            pred = cast(torch.Tensor, self.model(_obs))

            edge_mask = torch.cat(
                [(_s == self.horizon - 1).float(), torch.zeros((_s.shape[0], 1), device=self.device)], 1
            )
            logits = (pred[..., : self.ndim + 1] - inf * edge_mask).log_softmax(1)

            init_edge_mask = (_s == 0).float()
            back_logits = (pred[..., self.ndim + 1 :] - inf * init_edge_mask).log_softmax(1)

            logits = logits[:-1, :].gather(1, _a).squeeze(1)
            back_logits = (
                back_logits[1:-1, :].gather(1, _a[:-1, :]).squeeze(1)
                if _a[-1] == self.ndim
                else back_logits[1:, :].gather(1, _a).squeeze(1)
            )

            sum_logits = torch.sum(logits)
            sum_back_logits = torch.sum(back_logits)

            curr_ll_diff = self.log_Z + sum_logits - (_r + 1e-8).log() * self.beta - sum_back_logits
            loss += curr_ll_diff**2

        loss /= len(train_batch)
        return loss, {}


class CondTBAgent(BaseAgent):
    def __init__(
        self,
        envs: List[GridEnv],
        horizon: int,
        ndim: int,
        eps: float = 0.1,
        logit_temp: float = 1.0,
        device: Union[torch.device, str] = "cpu",
        use_buffer: bool = False,
        buffer_params: Optional[dict] = None,
        n_hidden: int = 256,
        n_layers: int = 2,
        cond_type: str = "logit",
    ) -> None:
        super().__init__(envs, horizon, ndim, eps, logit_temp, device, use_buffer, buffer_params)
        self.cond_type = cond_type
        
        out_dim = 2 * ndim + 1  # x 2 for forward/backward, + 1 for terminating action
        # self.model = MLP(dim_list=[horizon * ndim] + [n_hidden] * n_layers + [out_dim])
        # self.model.to(device)
        
        # self.beta_to_emb = MLP(dim_list=[1] + [n_hidden] + [n_hidden])
        # self.beta_to_emb.to(device)
        if self.cond_type == "layer":
            self.beta_to_emb = MLP(dim_list=[1] + [n_hidden] + [n_hidden])
            self.beta_to_emb.to(device)
            self.model = MLP(dim_list=[horizon * ndim + n_hidden] + [n_hidden] * n_layers + [out_dim])
            self.model.to(device)
        elif self.cond_type == "logit":
            self.beta_to_emb = MLP(dim_list=[1] + [n_hidden] + [n_hidden // 32])
            self.beta_to_emb.to(device)
            self.model = MLP(dim_list=[horizon * ndim] + [n_hidden] * n_layers + [out_dim])
            self.model.to(device)
            self.emb_to_temp = MLP(dim_list=[n_hidden // 32] + [n_hidden] + [1])
            self.emb_to_temp.to(device)
        elif self.cond_type == "logit+layer":
            self.beta_to_emb = MLP(dim_list=[1] + [n_hidden] + [n_hidden // 32])
            self.beta_to_emb.to(device)
            self.model = MLP(dim_list=[horizon * ndim + n_hidden // 32] + [n_hidden] * n_layers + [out_dim])
            self.model.to(device)
            self.emb_to_temp = MLP(dim_list=[n_hidden // 32] + [n_hidden] + [1])
            self.emb_to_temp.to(device)

        self.log_Z = MLP(dim_list=[1] + [n_hidden] + [1])
        self.log_Z.to(device)
        self.device = device
        
    def sample_many(self, batch_size: int, eval: bool = False, beta=1.0) -> Tuple[List[Tuple[torch.Tensor, ...]], List[Tuple]]:
        # replayed = self.replay.sample() if self.use_buffer and self.replay else []
        # TODO: Use the replay buffer

        assert batch_size <= len(self.envs)

        # Save trajectory
        batch_traj_obs, batch_traj_s, batch_traj_a = (
            [[] for _ in range(batch_size)],
            [[] for _ in range(batch_size)],
            [[] for _ in range(batch_size)],
        )

        # Save step results
        batch_obs, batch_r, batch_done, batch_s = [list(x) for x in zip(*[e.reset() for e in self.envs[:batch_size]])]
        batch_obs = self.to_ft(batch_obs)
        batch_s = self.to_lt(batch_s)
        if eval:
            batch_beta = torch.ones((len(batch_r), 1)).float().to(self.device) * beta
        else:
            batch_beta = torch.from_numpy(np.random.uniform(1.0, 3.0, size=(len(batch_r), 1))).float().to(self.device)
        batch_beta_orig = torch.clone(batch_beta)
        visited_x = []
        not_done_indices = [i for i in range(batch_size)]
        while not all(batch_done):
            with torch.no_grad():
                if self.cond_type == "layer":
                    batch_emb = self.beta_to_emb(batch_beta)
                    pred = cast(torch.Tensor, self.model(torch.cat([batch_obs, batch_emb], dim=-1)))
                elif self.cond_type == "logit":
                    pred = cast(torch.Tensor, self.model(batch_obs))
                    batch_temp = F.softplus(self.emb_to_temp(self.beta_to_emb(batch_beta)))
                    pred = torch.clamp(pred * batch_temp, -20.0, 20.0)
                elif self.cond_type == "logit+layer":
                    batch_emb = self.beta_to_emb(batch_beta)
                    pred = cast(torch.Tensor, self.model(torch.cat([batch_obs, batch_emb], dim=-1)))
                    batch_temp = F.softplus(self.emb_to_temp(self.beta_to_emb(batch_beta)))
                    pred = torch.clamp(pred * batch_temp, -20.0, 20.0)
                else:
                    raise NotImplementedError
                logit_temp = 1.0 if eval else self.logit_temp
                f_probs = (pred[..., : self.ndim + 1] / logit_temp).softmax(1)
                eps = 0.0 if eval else self.eps
                f_probs = (1 - eps) * f_probs + (eps) / (self.ndim + 1)
                actions = f_probs.multinomial(1)

            batch_step = [self.envs[i].step(int(a)) for i, a in zip(not_done_indices, actions)]

            for i, (curr_obs, curr_s, curr_a) in enumerate(zip(batch_obs, batch_s, actions)):
                env_idx = not_done_indices[i]
                batch_traj_obs[env_idx].append(curr_obs)
                batch_traj_s[env_idx].append(curr_s)
                batch_traj_a[env_idx].append(curr_a)

            new_batch_obs, new_batch_s, new_not_done_indices, new_batch_beta = [], [], [], []
            for i, (_obs, _r, _d, _s) in enumerate(batch_step):
                env_idx = not_done_indices[i]
                batch_done[env_idx] = _d

                if _d:
                    batch_r[env_idx] = _r
                    batch_traj_obs[env_idx].append(self.to_ft(_obs))
                    batch_traj_s[env_idx].append(self.to_lt(_s))
                    visited_x.append(tuple(_s))
                else:
                    new_batch_obs.append(_obs)
                    new_batch_s.append(_s)
                    new_not_done_indices.append(env_idx)
                    new_batch_beta.append(batch_beta[i])

            batch_obs, batch_s = self.to_ft(new_batch_obs), self.to_lt(new_batch_s)
            not_done_indices = new_not_done_indices
            batch_beta = torch.FloatTensor(new_batch_beta).to(self.device).unsqueeze(-1)

        batch_traj_obs_t = list(map(torch.stack, batch_traj_obs))
        batch_traj_s_t = list(map(torch.stack, batch_traj_s))
        batch_traj_a_t = list(map(torch.stack, batch_traj_a))
        batch_r = list(map(self.to_ft, batch_r))

        train_batch = list(zip(batch_traj_obs_t, batch_traj_s_t, batch_traj_a_t, batch_r, batch_beta_orig))

        return train_batch, visited_x
    
    def learn_from(self, train_batch: List[Tuple[torch.Tensor, ...]]) -> Tuple[torch.Tensor, Dict]:
        inf = 1000000000

        loss = torch.tensor(0.0, dtype=torch.float, device=self.device)
        for _obs, _s, _a, _r, _beta in train_batch:
            _beta = _beta.unsqueeze(-1)
            # print(_obs.shape, _s.shape, _a.shape, _r.shape)
            # _beta = torch.from_numpy(np.random.uniform(1.0, 3.0, size=(1, 1))).float().to(self.device).repeat(_obs.shape[0], 1)
            if self.cond_type == "layer":
                _emb = self.beta_to_emb(_beta.repeat(_obs.shape[0], 1))
                pred = cast(torch.Tensor, self.model(torch.cat([_obs, _emb], dim=-1)))
            elif self.cond_type == "logit":
                pred = cast(torch.Tensor, self.model(_obs))
                temp = F.softplus(self.emb_to_temp(self.beta_to_emb(_beta)))
                pred = torch.clamp(pred * temp, -20.0, 20.0)
            elif self.cond_type == "logit+layer":
                _emb = self.beta_to_emb(_beta.repeat(_obs.shape[0], 1))
                pred = cast(torch.Tensor, self.model(torch.cat([_obs, _emb], dim=-1)))
                temp = F.softplus(self.emb_to_temp(self.beta_to_emb(_beta)))
                pred = torch.clamp(pred * temp, -20.0, 20.0)

            edge_mask = torch.cat(
                [(_s == self.horizon - 1).float(), torch.zeros((_s.shape[0], 1), device=self.device)], 1
            )
            logits = (pred[..., : self.ndim + 1] - inf * edge_mask).log_softmax(1)

            init_edge_mask = (_s == 0).float()
            back_logits = (pred[..., self.ndim + 1 :] - inf * init_edge_mask).log_softmax(1)

            logits = logits[:-1, :].gather(1, _a).squeeze(1)
            back_logits = (
                back_logits[1:-1, :].gather(1, _a[:-1, :]).squeeze(1)
                if _a[-1] == self.ndim
                else back_logits[1:, :].gather(1, _a).squeeze(1)
            )

            sum_logits = torch.sum(logits)
            sum_back_logits = torch.sum(back_logits)

            curr_ll_diff = self.log_Z(_beta).sum() + sum_logits - (_r + 1e-8).log() * _beta[0].item() - sum_back_logits
            loss += curr_ll_diff**2

        loss /= len(train_batch)
        return loss, {}
