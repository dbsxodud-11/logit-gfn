"""
Trainer for the GridWorld environment
"""

import sys
sys.path.append("..")

from collections import defaultdict
from copy import deepcopy
from typing import Callable, Dict, List, Optional, Tuple
import random
import itertools

from tqdm import tqdm
import numpy as np
import torch
torch.set_num_threads(1)
from torch.nn.utils.clip_grad import clip_grad_norm_
from torch.optim import Adam

import io
from PIL import Image
import matplotlib.pyplot as plt

from grid.env import GridEnv
from grid.models import BaseAgent, TBAgent, CondTBAgent

def seed_everything(seed: Optional[int] = None) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)  # type: ignore
    torch.backends.cudnn.deterministic = True  # type: ignore
    torch.backends.cudnn.benchmark = True  # type: ignore


def fig2img(fig):
    """Convert a Matplotlib figure to a PIL Image and return it"""
    buf = io.BytesIO()
    fig.savefig(buf)
    buf.seek(0)
    img = Image.open(buf)
    return img

class Trainer:
    def __init__(
        self,
        env: GridEnv,
        agent: BaseAgent,
        lr_dict: Dict[str, float],
        rnd_train_interval: int = 1,
        clip_grad_norm: float = 1.0,
        n_train_steps: int = 20000,
        batch_size: int = 16,
        ns_batch_size: int = 8,
        ts_ratio: float = 1,
        no_tqdm: bool = False,
        num_empirical_loss: int = 200000,
        logging_fn: Optional[Callable] = None,  # wandb
        eval_interval: int = 1000,
        eval_num_empirical_loss: int = 10000,
        eval_batch_size: int = 1000,
        plot: bool = False,
        cond_type: str = "layer",
    ) -> None:
        self.env = env
        self.agent = agent

        # Optimizer
        if isinstance(self.agent, TBAgent):  # Z for TB requires higher lr
            self.params_list = [
                {"params": self.agent.model.parameters(), "lr": lr_dict["tb"]},
                {"params": [self.agent.log_Z], "lr": lr_dict["tb_z"]}
            ]
        elif isinstance(self.agent, CondTBAgent):
            if cond_type == "layer":
                self.params_list = [
                    {"params": self.agent.model.parameters(), "lr": lr_dict["tb"]},
                    {"params": self.agent.beta_to_emb.parameters(), "lr": lr_dict["tb"]},
                    {"params": self.agent.log_Z.parameters(), "lr": lr_dict["tb_z"]}
                ]
            elif cond_type == "logit":
                self.params_list = [
                    {"params": self.agent.model.parameters(), "lr": lr_dict["tb"]},
                    {"params": self.agent.beta_to_emb.parameters(), "lr": lr_dict["tb"]},
                    {"params": self.agent.emb_to_temp.parameters(), "lr": lr_dict["tb"]},
                    {"params": self.agent.log_Z.parameters(), "lr": lr_dict["tb_z"]}
                ]
            elif cond_type == "logit+layer":
                self.params_list = [
                    {"params": self.agent.model.parameters(), "lr": lr_dict["tb"]},
                    {"params": self.agent.beta_to_emb.parameters(), "lr": lr_dict["tb"]},
                    {"params": self.agent.emb_to_temp.parameters(), "lr": lr_dict["tb"]},
                    {"params": self.agent.log_Z.parameters(), "lr": lr_dict["tb_z"]}
                ]
            else:
                raise NotImplementedError
        else:
            raise ValueError(f"Unknown agent: {self.agent}")


        self.optimizer = Adam(self.params_list)
        self.clip_grad_norm: float = clip_grad_norm

        # training
        self.n_train_steps: int = n_train_steps
        self.batch_size: int = batch_size
        self.ts_ratio = max(int(ts_ratio), 1)  # train / sample ratio
        self.st_ratio = max(int(1 / ts_ratio), 1)  # sample / train ratio

        # logging
        self.no_tqdm: bool = no_tqdm
        self.num_empirical_loss: int = num_empirical_loss
        self.logging_fn = logging_fn

        # evaluation
        self.eval_interval = eval_interval
        self.eval_num_empirical_loss = eval_num_empirical_loss
        self.eval_batch_size = eval_batch_size
        self.plot = plot

    def train(self) -> None:
        # Plot empirical distribution if ndim == 2
        if self.plot:
            self.plot_target_distribution()

        losses = []
        infos = []
        all_visited = []
        main_visited = []  # for main policy
        ns_visited = []  # for NS-GFN
        # empirical_distrib_losses = []

        for i in tqdm(range(1, self.n_train_steps + 1), disable=self.no_tqdm):
            train_batch = []

            for _ in range(self.st_ratio):
                _batch, _visited = self.agent.sample_many(self.batch_size)

                train_batch += _batch
                all_visited += _visited

            for _ in range(self.ts_ratio):
                loss, info = self.agent.learn_from(train_batch)

                self.optimizer.zero_grad()
                loss.backward()
                if self.clip_grad_norm > 0:
                    clip_grad_norm_(itertools.chain(*[_l["params"] for _l in self.params_list]), self.clip_grad_norm)
                self.optimizer.step()

                losses.append(loss.item())
                if len(info) > 0:
                    infos.append(info)

            # Logging
            if not i % 100:
                l1, w_l1 = self.compute_empirical_distribution_error(all_visited[-self.num_empirical_loss :], beta=1.0)
                # empirical_distrib_losses.append((l1, kl))

                print(f"Step: {i}, Loss: {np.mean(losses[-100:]):.6f}")
                print(f"empirical L1 dist: {l1:.6f}, weighted L1 dist: {w_l1:.6f}, #visited: {len(all_visited)}")

                # if len(infos) > 0:
                #     print(f"[Last Info] {infos[-1]}")

                goal_list = self.env.goals
                goal_found_map = {goal: 0 for goal in goal_list}
                for s in all_visited:
                    if s in goal_found_map:
                        goal_found_map[s] += 1
                # print("goal_found", goal_found_map)
                print(f"n_goals_found: {sum([1 for v in goal_found_map.values() if v > 0])}/{len(goal_list)}")

                if self.logging_fn is not None:
                    self.logging_fn(
                        {
                            "loss": np.mean(losses[-100:]),
                            "empirical_l1": l1,
                            "empirical_w_l1": w_l1,
                            "n_goals_found": sum([1 for v in goal_found_map.values() if v > 0]),
                        },
                        step=i,
                    )

            # Sampling and Evaluation
            if not i % self.eval_interval:
                self.agent.eval()

                eval_visited_list = []
                vmax = 0
                for beta in [1, 2, 3, 4, 5]:
                    n_sampled = 0
                    eval_visited = []
                    eval_ns_visited = []
                    while n_sampled < self.eval_num_empirical_loss:
                        batch_size = min(self.eval_batch_size, self.eval_num_empirical_loss - n_sampled)
                        _, _eval_visited = self.agent.sample_many(batch_size, eval=True, beta=beta)
                        eval_visited += _eval_visited
                        n_sampled += batch_size

                    assert len(eval_visited) == self.eval_num_empirical_loss

                    eval_l1, eval_w_l1 = self.compute_empirical_distribution_error(eval_visited, beta)
                    print(f"Eval(beta={beta}): empirical L1 dist: {eval_l1:.6f}, weighted L1 dist: {eval_w_l1:.6f}, #visited: {self.eval_num_empirical_loss}")
                    if self.logging_fn is not None:
                        self.logging_fn({f"eval_empirical_l1_beta{beta}": eval_l1, f"eval_empirical_w_l1_beta{beta}": eval_w_l1,}, step=i)
                    eval_visited_list.append(eval_visited)

                # Plot empirical distribution
                if self.plot:
                    self.plot_visit_counts(all_visited, all_visited[-self.eval_interval * self.batch_size:], eval_visited, step=i)
                    self.plot_empirical_distribution(eval_visited_list, label=f"empirical_density_beta", step=i)

                self.agent.train()
        
        if isinstance(self.agent, CondTBAgent):
            torch.save(self.agent.state_dict(), f"models/model_d{self.env.ndim}_H{self.env.horizon}_{self.agent.cond_type}.pt")
        else:
            torch.save(self.agent.state_dict(), f"models/model_d{self.env.ndim}_H{self.env.horizon}_tb{self.agent.beta}.pt")
            

    def evaluate(self):
        eval_visited_list = []
        vmax = 0.0
        for i, beta in enumerate([1, 2, 3, 4, 5]):
            if isinstance(self.agent, CondTBAgent):
                self.agent.load_state_dict(torch.load(f"models/model_d{self.env.ndim}_H{self.env.horizon}_{self.agent.cond_type}.pt"))
            else:
                self.agent.load_state_dict(torch.load(f"models/model_d{self.env.ndim}_H{self.env.horizon}_tb{float(beta)}.pt"))
            
            self.agent.eval()
            
            n_sampled = 0
            eval_visited = []
            eval_ns_visited = []
            while n_sampled < self.eval_num_empirical_loss:
                batch_size = min(self.eval_batch_size, self.eval_num_empirical_loss - n_sampled)
                _, _eval_visited = self.agent.sample_many(batch_size, eval=True, beta=beta)
                eval_visited += _eval_visited
                n_sampled += batch_size

            assert len(eval_visited) == self.eval_num_empirical_loss

            eval_l1, eval_w_l1 = self.compute_empirical_distribution_error(eval_visited, beta)
            print(f"Eval(beta={beta}): empirical L1 dist: {eval_l1:.6f}, weighted L1 dist: {eval_w_l1:.6f}, #visited: {self.eval_num_empirical_loss}")
            if self.logging_fn is not None:
                self.logging_fn({f"eval_empirical_l1_beta{beta}": eval_l1, f"eval_empirical_w_l1_beta{beta}": eval_w_l1,}, step=i)
            eval_visited_list.append(eval_visited)

        # Plot empirical distribution
        if self.plot:
            self.plot_empirical_distribution(eval_visited_list, label=f"empirical_density_beta", step=i)
    
    
    def compute_empirical_distribution_error(self, visited: List[Tuple], beta=1.0) -> Tuple[float, float]:
        if not len(visited):
            return 100, 100

        hist = defaultdict(int)
        for s in visited:
            hist[s] += 1

        unnormalized_density, true_density, end_states = self.env.get_true_density(beta)
        Z = sum([hist[_s] for _s in end_states])
        estimated_density = np.array([hist[_s] / Z for _s in end_states], dtype=np.float32)

        # L1 distance
        l1 = abs(estimated_density - true_density).mean()
        # weighted L1 distance (weighted by unnormalized density)
        weighted_l1 = (abs(estimated_density - true_density) * unnormalized_density).mean()
        # # KL divergence
        # kl = (true_density * np.log(estimated_density / true_density)).sum()
        return l1, weighted_l1

    def plot_visit_counts(self, all_visited, this_visited, eval_visited, step=None):
        visit_count = np.zeros_like(self.env.all_rewards).reshape(*(self.env.horizon,) * self.env.ndim)
        for s in all_visited:
            visit_count[s] += 1

        this_visit_count = np.zeros_like(visit_count)
        for s in this_visited:
            this_visit_count[s] += 1
        
        imagined_visit_count = np.zeros_like(visit_count)
        for s in eval_visited:
            imagined_visit_count[s] += 1

        fig, axes = plt.subplots(1, 3, figsize=(21, 6))
        for dat, ax in zip([visit_count, this_visit_count, imagined_visit_count], axes):
            if dat.ndim != 2:
                if dat.ndim % 2 == 0:
                    power = dat.ndim // 2
                    dat = dat.reshape(*(self.env.horizon ** power,) * 2)
                else:
                    raise ValueError(f"array.ndim should be even number, but {dat.ndim}")
            img = ax.imshow(dat, origin="lower")
            fig.colorbar(img, ax=ax)
        if self.logging_fn is not None:
            image = wandb.Image(fig)
            self.logging_fn({"all_vs_this_vs_imagined": image}, step=step)
        plt.close(fig)

    def plot_empirical_distribution(self, visited_list, label="empirical_density", step=None):
        empirical_densities = []
        vmax = 0.0
        for i, visited in enumerate(visited_list):
            empirical_density = np.zeros_like(self.env.all_rewards).reshape(*(self.env.horizon,) * self.env.ndim)
            for s in visited:
                empirical_density[s] += 1
            empirical_density = empirical_density / empirical_density.sum()
            
            empirical_densities.append(empirical_density)
            vmax = max(vmax, empirical_density.max())
        for i, beta in enumerate([1, 2, 3, 4, 5]):
            self.plot_fn(empirical_densities[i], label=f"{label}_beta{beta}", vmax=vmax, step=step)
            
        fig, ax = plt.subplots(1, 5, figsize=(24, 6))
        for i, beta in enumerate([1, 2, 3, 4, 5]):
            img = ax[i].imshow(empirical_densities[i], origin="lower", vmax=vmax)
            
        # fig.subplots_adjust(right=0.85)
        cb_ax = fig.add_axes([0.92, 0.15, 0.01, 0.7])
        fig.colorbar(img, cax=cb_ax)
        if self.logging_fn is not None:
            # image = wandb.Image(fig2img(fig))
            self.logging_fn({label: wandb.Image(fig)}, step=step)
        plt.close(fig)

    def plot_target_distribution(self):
        target_densities = []
        vmax = 0.0
        for beta in [1, 2, 3, 4, 5]:
            target_density = self.env.all_rewards.reshape(*(self.env.horizon,) * self.env.ndim) ** beta
            target_density = target_density / target_density.sum()
            
            target_densities.append(target_density)
            vmax = max(vmax, target_density.max())
        for i, beta in enumerate([1, 2, 3, 4, 5]):
            self.plot_fn(target_densities[i], label=f"target_density_beta{beta}", vmax=vmax, step=0)
            
        fig, ax = plt.subplots(1, 5, figsize=(24, 6))
        for i, beta in enumerate([1, 2, 3, 4, 5]):
            img = ax[i].imshow(target_densities[i], origin="lower", vmax=vmax)
            
        # fig.subplots_adjust(right=0.85)
        cb_ax = fig.add_axes([0.92, 0.15, 0.01, 0.7])
        fig.colorbar(img, cax=cb_ax)
        if self.logging_fn is not None:
            # image = wandb.Image(fig2img(fig))
            self.logging_fn({"target_density": wandb.Image(fig)}, step=0)
        plt.close(fig)

    def plot_fn(self, array, label, vmax=None, step=None, show=False):
        if array.ndim != 2:
            if array.ndim % 2 == 0:
                power = array.ndim // 2
                array = array.reshape(*(self.env.horizon ** power,) * 2)
            else:
                raise ValueError(f"array.ndim should be even number, but {array.ndim}")

        fig, ax = plt.subplots(1, 1, figsize=(7, 6))
        img = ax.imshow(array, origin="lower", vmax=vmax)
        plt.colorbar(img, ax=ax)
        if self.logging_fn is not None:
            # image = wandb.Image(fig2img(fig))
            self.logging_fn({label: wandb.Image(fig)}, step=step)
        if show:
            plt.show()
        plt.close(fig)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    
    parser.add_argument("--run_name", type=str, default="")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--device_num", type=int, default=0)

    # Env
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--ndim", type=int, default=2)
    parser.add_argument(
        "--reward_type", type=str, default="dense", choices=["dense", "sparse"]
    )
    parser.add_argument("--beta", type=float, default=1.0)

    # Agent
    parser.add_argument("--agent", type=str, default="tb", choices=["tb", "condtb"])
    parser.add_argument("--eps", type=float, default=0.0)  # epsilon for exploration
    parser.add_argument("--logit_temp", type=float, default=1.0)  # logit temp for exploration-exploitation trade-off
    parser.add_argument("--cond_type", type=str, default="none")
    parser.add_argument("--tb_beta", type=float, default=1.0)

    # FMFlowNetAgent
    parser.add_argument("--n_hidden", type=int, default=256)
    parser.add_argument("--n_layers", type=int, default=2)
    parser.add_argument("--use_buffer", action="store_true")

    # Buffer
    parser.add_argument("--buffer_size", type=int, default=100)
    parser.add_argument("--sample_size", type=int, default=2)
    parser.add_argument("--add_strategy", type=str, default="top_k")
    parser.add_argument("--sample_strategy", type=str, default="random")

    # Logger
    parser.add_argument("--logger", type=str, default="none", choices=["none", "wandb"])

    # Trainer
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--tb_z_lr", type=float, default=1e-1)
    parser.add_argument("--clip_grad_norm", type=float, default=1.0)
    parser.add_argument("--n_train_steps", type=int, default=-1)  # 10000 for H<=256 for 2D and H<=16 for 4D // 20000 otherwise
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--ts_ratio", type=float, default=1)
    parser.add_argument("--no_tqdm", action="store_true")
    parser.add_argument("--num_empirical_loss", type=int, default=-1)  # 100000 for H<=256 for 2D and H<=16 for 4D // 200000 otherwise

    # Evaluation
    parser.add_argument("--eval_interval", type=int, default=1000)
    parser.add_argument("--eval_num_empirical_loss", type=int, default=10000)
    parser.add_argument("--eval_batch_size", type=int, default=100)
    parser.add_argument("--plot", action="store_true")

    args = parser.parse_args()

    if int(args.n_train_steps) == -1:
        if args.ndim == 2:
            args.n_train_steps = 10000 if args.horizon <= 256 else 20000
        elif args.ndim == 4:
            args.n_train_steps = 10000 if args.horizon <= 16 else 20000
        else:
            args.n_train_steps = 20000
    if int(args.num_empirical_loss) == -1:
        if args.ndim == 2:
            args.num_empirical_loss = 100000 if args.horizon <= 256 else 200000
        elif args.ndim == 4:
            args.num_empirical_loss = 100000 if args.horizon <= 16 else 200000
        else:
            args.num_empirical_loss = 200000
    if args.plot:
        assert args.logger == "wandb", "plot option is only available when logger is wandb"

    # Set seed
    seed_everything(args.seed)

    # Set device
    device = torch.device(f"cuda:{args.device_num}" if args.device == "cuda" and torch.cuda.is_available() else "cpu")  # type: ignore

    env_params = {
        "horizon": args.horizon,
        "ndim": args.ndim,
        "reward_type": args.reward_type,
        "beta": args.beta,
    }
    env = GridEnv(**env_params)

    agent_common_params = {
        "envs": [deepcopy(env) for _ in range(max(args.batch_size, args.eval_batch_size))],
        "horizon": args.horizon,
        "ndim": args.ndim,
        "eps": args.eps,
        "logit_temp": args.logit_temp,
        "device": device,
        "use_buffer": args.use_buffer,
        "buffer_params": {
            "buffer_size": args.buffer_size,
            "sample_size": args.sample_size,
            "add_strategy": args.add_strategy,
            "sample_strategy": args.sample_strategy,
        },
    }
    
    agent_dict = {
        "tb": (
            TBAgent,
            {
                "n_hidden": args.n_hidden,
                "n_layers": args.n_layers,
                "tb_beta": args.tb_beta,
            },
        ),
        "condtb": (
            CondTBAgent,
            {
                "n_hidden": args.n_hidden,
                "n_layers": args.n_layers,
                "cond_type": args.cond_type,
            }
        )
    }

    agent_class, agent_param = agent_dict[args.agent]
    agent = agent_class(**agent_common_params, **agent_param)

    lr_dict = {
        "tb": args.lr,
        "tb_z": args.tb_z_lr,
    }

    trainer_params = {
        "lr_dict": lr_dict,
        "clip_grad_norm": args.clip_grad_norm,
        "n_train_steps": args.n_train_steps,
        "batch_size": args.batch_size,
        "ts_ratio": args.ts_ratio,
        "no_tqdm": args.no_tqdm,
        "num_empirical_loss": args.num_empirical_loss,
        "eval_interval": args.eval_interval,
        "eval_num_empirical_loss": args.eval_num_empirical_loss,
        "eval_batch_size": args.eval_batch_size,
        "plot": args.plot,
        "cond_type": args.cond_type,
    }

    logging_fn = None
    if args.logger == "wandb":
        import wandb

        wandb.init(
            project="tempgfn-grid",
            name=f"{args.reward_type}_{args.agent}_h{args.horizon}n{args.ndim}s{args.seed}{'_'+args.run_name if args.run_name else ''}",
        )
        for config in [env_params, agent_common_params, agent_param, trainer_params]:
            wandb.config.update(config)
        wandb.config.update({"env_param_str": f"h{args.horizon}ndim{args.ndim}", "agent": args.agent, "run_name": args.run_name})
        wandb.watch(agent)
        logging_fn = wandb.log

    trainer = Trainer(env, agent, logging_fn=logging_fn, **trainer_params)
    trainer.train()
    # trainer.evaluate()
