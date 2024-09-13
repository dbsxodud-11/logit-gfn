"""
GridWorld environment
"""

from typing import Callable, List, Optional, Tuple, Union, cast
import itertools

import numpy as np

###################
# Reward function #
###################


class GridDenseCornerReward:
    def __init__(self, horizon: int, beta: float, *args, **kwargs):
        self.horizon = horizon
        self.beta = beta

    def func(self, x: np.ndarray) -> np.ndarray:
        normalized_x = x / (self.horizon - 1) * 2 - 1
        abs_x = np.abs(normalized_x)
        rewards = (abs_x > 0.5).prod(-1) * 0.5 + ((abs_x < 0.8) * (abs_x > 0.6)).prod(-1) * 2 + 1e-1
        rewards = rewards ** self.beta
        return rewards

class GridSparseCornerReward:
    def __init__(self, horizon: int, ndim: int, beta: float, *args, **kwargs):
        self.horizon = horizon
        if ndim == 2:
            self.targets = np.array([[horizon - 2, horizon - 2], [horizon - 2, 1], [1, horizon - 2]])
        else:
            self.targets = np.array(list(set(itertools.permutations([1] * (ndim // 2) + [horizon - 2] * (ndim // 2)))))
        self.beta = beta

    def func(self, x: np.ndarray):
        if len(x.shape) == 1:
            rewards = (x[None, :] == self.targets).prod(-1).sum(-1)
        else:
            rewards = (x[:, None, :] == self.targets[None, :, :]).prod(-1).sum(-1)

            # # save reward map as image
            # import matplotlib.pyplot as plt
            # plt.imshow(rewards.reshape(self.horizon, self.horizon), origin="lower")
            # plt.savefig(f"sparse_reward_map{self.horizon}{self.horizon}.png")
            # raise

            # # save reward map as image
            # import matplotlib.pyplot as plt
            # plt.imshow(rewards.reshape(self.horizon ** 2, self.horizon ** 2), origin="lower")
            # plt.savefig(f"sparse_reward_map{self.horizon}{self.horizon}{self.horizon}{self.horizon}.png")
            # raise

        rewards = rewards ** self.beta

        return rewards


def get_reward_func(reward_type: str, horizon: int, ndim: int, beta: float, *args, **kwargs) -> Callable:
    if reward_type == "dense":
        return GridDenseCornerReward(horizon, beta, *args, **kwargs).func
    if reward_type == "sparse":
        return GridSparseCornerReward(horizon, ndim, beta, *args, **kwargs).func
    raise NotImplementedError


###################
#     GridEnv     #
###################


class GridEnv:
    """
    GridEnv with option for sparse reward setting
    reference: https://github.com/ling-pan/GAFN
    """

    def __init__(self, horizon: int, ndim: int, reward_type: str, beta: float) -> None:
        self.horizon = horizon
        self.ndim = ndim
        self.reward_type = reward_type
        self.reward_func = get_reward_func(reward_type, horizon, ndim, beta)

        self.true_density_info = None
        self.all_int_states = np.array(list(itertools.product(*[list(range(self.horizon))] * self.ndim)), dtype=np.int32)

        self.all_rewards = self.reward_func(self.all_int_states)
        reward_max = self.all_rewards.max() if reward_type != "fourier" else np.quantile(self.all_rewards, 0.95)
        self.goals = list(map(tuple, self.all_int_states[self.all_rewards >= reward_max]))

    def obs(self, state: Optional[np.ndarray] = None) -> np.ndarray:
        # state: (ndim,) with dtype=np.int32
        state = self.state if state is None else state
        z = np.zeros((self.horizon * self.ndim), dtype=np.float32)
        z[np.arange(self.ndim) * self.horizon + state] = 1
        return z

    def reset(self) -> Tuple[np.ndarray, np.ndarray, bool, np.ndarray]:
        self.state = np.zeros(self.ndim, dtype=np.int32)
        return self.obs(self.state), self.reward_func(self.state), False, self.state

    def step(
        self, action: int, state: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, Union[float, np.ndarray], bool, np.ndarray]:
        s = cast(np.ndarray, state or self.state)

        if action < self.ndim:
            s[action] += 1
        done = s.max() >= self.horizon - 1 or action == self.ndim
        return self.obs(s), 0 if not done else self.reward_func(s), done, s

    def parent_transitions(
        self, state: np.ndarray, used_stop_action: bool = False
    ) -> Tuple[List[np.ndarray], List[int]]:
        if used_stop_action:
            return [self.obs(state)], [self.ndim]

        parents = []
        actions = []
        for i in range(self.ndim):
            if state[i] > 0:
                state_parent = state.copy()
                state_parent[i] -= 1
                if state_parent.max() == self.horizon - 1:  # can't have a terminal parent
                    continue
                parents += [self.obs(state_parent)]  # Note that parents are saved as obs
                actions += [i]
        return parents, actions

    def get_true_density(self, beta=1.0) -> Tuple[np.ndarray, np.ndarray, List]:
        # These calculation doesn't have to be done on GPU
        if self.true_density_info is not None:
            return self.true_density_info

        state_mask = np.array(
            [len(self.parent_transitions(s)[0]) > 0 or sum(s) == 0 for s in self.all_int_states], dtype=np.bool_
        )
        unnormalized_density = self.all_rewards[state_mask] ** beta
        true_density_info = (
            unnormalized_density,  # unnormalized density
            unnormalized_density / unnormalized_density.sum(),  # normalized density
            list(map(tuple, self.all_int_states[state_mask])),  # keys (states)
        )
        return true_density_info
