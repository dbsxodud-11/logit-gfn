'''
    GFP
    Transformer Proxy
    Start from scratch
'''

import copy, pickle, functools
import numpy as np
from tqdm import tqdm
import pandas as pd
import torch
from polyleven import levenshtein

import gflownet.trainers as trainers
from gflownet.GFNs import models
from gflownet.MDPs import seqpamdp, seqinsertmdp, seqarmdp
from gflownet.monitor import TargetRewardDistribution, Monitor, diversity

import flexs
from flexs import baselines
import flexs.utils.sequence_utils as s_utils

def dynamic_inherit_mdp(base, args):

  class RNAMDP(base):
    def __init__(self, args):
      super().__init__(args,
                       alphabet=["U", "C", "G", "A"],
                       forced_stop_len=args.forced_stop_len)
      self.args = args
      
      print(f'Loading data ...')
      munge = lambda x: ''.join([str(c) for c in list(x)])
      problem = flexs.landscapes.rna.registry()['L14_RNA1']
      self.proxy_model = flexs.landscapes.RNABinding(**problem['params'])
      
      allpreds_file = args.allpreds_file
      with open(allpreds_file, 'rb') as f:
        self.rewards = pickle.load(f)
      print(problem)
      
      # scale rewards
      py = np.array(list(self.rewards))

      self.SCALE_REWARD_MIN = args.scale_reward_min
      self.SCALE_REWARD_MAX = args.scale_reward_max
      self.REWARD_EXP = args.reward_exp
      self.REWARD_MAX = max(py)

      py = np.maximum(py, self.SCALE_REWARD_MIN)
      py = py ** self.REWARD_EXP
      self.scale = self.SCALE_REWARD_MAX / max(py)
      py = py * self.scale
      
      self.scaled_rewards = py
      
      # modes
      mode_file = args.mode_file
      with open(mode_file, 'rb') as f:
        self.modes = pickle.load(f)
      print(f"Found num modes: {len(self.modes)}")
      
      # offline dataset
      if args.task == "offline_generalization":
        offline_xs = np.load(args.offline_xs_file)
        offline_ys = np.load(args.offline_ys_file)
        offline_ys_norm = (np.maximum(offline_ys, self.SCALE_REWARD_MIN) ** self.REWARD_EXP) * self.scale 
        self.offline_dataset = {self.state(munge("".join(map(lambda x_elem: self.alphabet[x_elem], x))), is_leaf=True): float(y)
                                for x, y in zip(offline_xs, offline_ys_norm)}
        print(f"Dataset Size: {len(self.offline_dataset)}\tDataset Max: {offline_ys.max().item()}")
        
    # Core
    def reward(self, x):
      assert x.is_leaf, 'Error: Tried to compute reward on non-leaf node.'
      r = self.proxy_model.get_fitness([x.content]).item()
      
      r = np.maximum(r, self.SCALE_REWARD_MIN)
      r = r ** self.REWARD_EXP
      r = r * self.scale
      return r

    def is_mode(self, x, r):
      return x.content in self.modes
    
    def unnormalize(self, r):
      r = r / self.scale
      r = r ** (1 / self.REWARD_EXP)
      return r

    '''
      Interpretation & visualization
    '''
    def dist_func(self, state1, state2):
      """ States are SeqPAState or SeqInsertState objects. """
      return levenshtein(state1.content, state2.content)

    def make_monitor(self):
      target = TargetRewardDistribution()
      target.init_from_base_rewards(self.rewards, target_beta=self.args.target_beta)
      return Monitor(self.args, target, dist_func=self.dist_func,
                    is_mode_f=self.is_mode,
                    unnormalize=self.unnormalize)
    
    def reduce_storage(self):
      del self.rewards
      del self.scaled_rewards

  return RNAMDP(args)


def offline_generalization(args):
  print("Offline generalization in rna ...")
  base = seqpamdp.SeqPrependAppendMDP
  actorclass = seqpamdp.SeqPAActor
  mdp = dynamic_inherit_mdp(base, args)
  
  actor = actorclass(args, mdp)
  model = models.make_model(args, mdp, actor)
  monitor = mdp.make_monitor()
  
  mdp.reduce_storage()
  
  trainer = trainers.Trainer(args, model, mdp, actor, monitor)
  trainer.offline_generalization(offline_dataset=mdp.offline_dataset)

def mode_seeking(args):
  print("Online mode seeking in RNA-Binding...")
  base = seqpamdp.SeqPrependAppendMDP
  actorclass = seqpamdp.SeqPAActor
  mdp = dynamic_inherit_mdp(base, args)
  
  actor = actorclass(args, mdp)
  model = models.make_model(args, mdp, actor)
  monitor = mdp.make_monitor()
  
  mdp.reduce_storage()
  
  trainer = trainers.Trainer(args, model, mdp, actor, monitor)
  trainer.mode_seeking()
  return
