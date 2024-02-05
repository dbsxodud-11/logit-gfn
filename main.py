'''
  Run experiment with wandb logging.

  Usage:
  python runexpwb.py --setting bag

  Note: wandb isn't compatible with running scripts in subdirs:
    e.g., python -m exps.chess.chessgfn
  So we call wandb init here.
'''
import argparse
import random
import torch
import wandb
import options
import numpy as np
from attrdict import AttrDict

from exps.qm9str import qm9str
from exps.sehstr import sehstr
from exps.tfbind8 import tfbind8_oracle
from exps.rna import rna

mode_seeking = {
  'qm9str': lambda args: qm9str.mode_seeking(args),
  'sehstr': lambda args: sehstr.mode_seeking(args),
  'tfbind8': lambda args: tfbind8_oracle.mode_seeking(args),
  'rna': lambda args: rna.mode_seeking(args),
}

offline_generalization = {
  'qm9str': lambda args: qm9str.offline_generalization(args),
  'sehstr': lambda args: sehstr.offline_generalization(args),
  'tfbind8': lambda args: tfbind8_oracle.offline_generalization(args),
  'rna': lambda args: rna.offline_generalization(args),
}

training_stability = {
  'qm9str': lambda args: qm9str.training_stability(args),
  'sehstr': lambda args: sehstr.training_stability(args),
  'tfbind8': lambda args: tfbind8_oracle.training_stability(args),
  'rna': lambda args: rna.training_stability(args),
}

def main(args):
  print(f"Task: {args.task}")
  print(f"Setting: {args.setting}")
  
  if args.task == "mode_seeking":
    exp_f = mode_seeking[args.setting]
    exp_f(args)
  elif args.task == "offline_generalization":
    exp_f = offline_generalization[args.setting]
    exp_f(args)
  return

def set_seed(seed=0):
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)


if __name__ == '__main__':
  args = options.parse_args()
  set_seed(args.seed)
  
  wandb.init(project=args.wandb_project,
             entity=args.wandb_entity,
             config=args, 
             mode=args.wandb_mode)
  args = AttrDict(wandb.config)
  
  run_name = args.task + "/"
  if args.temp_cond:
    if args.temp_cond_type == "logit" and args.layer_conditioning:
      # Layer-conditioning for Logit-GFN
      run_name += "logit+layer" + "_" + "gfn"
    elif args.temp_cond_type == "layer" and args.thermometer:
      # Layer-GFN with thermometer embedding
      run_name += args.temp_cond_type + "_" + "thermo" + "_" + "gfn"
    else:
      run_name += args.temp_cond_type + "_" + "gfn"
  else:
    run_name += "gfn"
    
  run_name += "_" + args.loss_type
  if args.loss_type == "subtb":
    run_name += f"{float(args.lamda)}"
  run_name += "_" + f"k{args.num_steps_per_batch}"
  
  if args.temp_cond:
    if args.train_temp_dist == "constant":
      run_name += "/" + f"train_dist_{args.train_temp_dist}_{float(args.train_temp)}"
    elif args.train_temp_dist == "normal":
      run_name += "/" + f"train_dist_{args.train_temp_dist}_{float(args.train_temp_mu)}-{float(args.train_temp_sigma)}"
    else:
      run_name += "/" + f"train_dist_{args.train_temp_dist}_{float(args.train_temp_min)}-{float(args.train_temp_max)}"
      
    if args.exp_temp_dist == "constant":
      run_name += "/" + f"exp_dist_{args.exp_temp_dist}_{float(args.exp_temp)}"
    elif args.exp_temp_dist == "normal":
      run_name += "/" + f"exp_dist_{args.exp_temp_dist}_{float(args.exp_temp_mu)}-{float(args.exp_temp_sigma)}"
    else:
      run_name += "/" + f"exp_dist_{args.exp_temp_dist}_{float(args.exp_temp_min)}-{float(args.exp_temp_max)}" 
  else:
    run_name += "/" + f"target_{float(args.target_beta)}"
    
  run_name += "/" + f"seed_{args.seed}"
  args.run_name = run_name.upper()

  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  print(f'device={device}')
  args.device = device
 
  main(args)
