import random
import pickle
import numpy as np
import torch
from tqdm import tqdm

from .data import Experience
from .conditioning import TemperatureConditional


class Trainer:
  def __init__(self, args, model, mdp, actor, monitor):
    self.args = args
    self.model = model
    self.mdp = mdp
    self.actor = actor
    self.monitor = monitor
    
  def offline_generalization(self, offline_dataset):
    allXtoR = offline_dataset
    
    num_offline = self.args.num_offline_batches_per_round
    offline_bsize = self.args.num_samples_per_offline_batch
    monitor_fast_every = self.args.monitor_fast_every
    monitor_num_samples = self.args.monitor_num_samples
    
    if self.args.temp_cond:
      temp_sampler = TemperatureConditional(self.args)
      
    print(f'Starting training. \
            Each round: num_offline={num_offline}')

    for round_num in tqdm(range(self.args.num_offline_training_rounds)):
      for _ in range(num_offline):
        offline_xs = self.select_offline_xs(allXtoR, offline_bsize)
        if self.args.temp_cond:
          temp = temp_sampler.sample(offline_bsize, exp=False,
                                                   fraction=round_num / self.args.num_offline_training_rounds)
          offline_dataset = self.offline_PB_traj_sample(offline_xs, allXtoR, temp=temp["beta_enc"])
          for _ in range(self.args.num_steps_per_batch):
            self.model.train(offline_dataset, temp=temp, online=False)
        else:
          offline_dataset = self.offline_PB_traj_sample(offline_xs, allXtoR)
          for _ in range(self.args.num_steps_per_batch):
            self.model.train(offline_dataset, online=False)
            
      # monitor
      if round_num % monitor_fast_every == 0 and round_num > 0:
        if self.args.temp_cond:
          temp = temp_sampler.sample(monitor_num_samples, exp=True, 
                                                          fraction=round_num / self.args.num_offline_training_rounds)
          truepolicy_data = self.model.batch_fwd_sample(monitor_num_samples,
                                                        temp=temp["beta_enc"],
                                                        epsilon=0.0)
        else:
          truepolicy_data = self.model.batch_fwd_sample(monitor_num_samples,
                                                        epsilon=0.0)
        self.monitor.log_samples(round_num, truepolicy_data)
      self.monitor.maybe_eval_samplelog(self.model, round_num, allXtoR)
      
      # save    
      if round_num and round_num % self.args.save_every_x_active_rounds == 0:
        self.model.save_params(self.args.save_models_dir + \
                               self.args.run_name + "/" + f'round_{round_num}.pth')
    print('Finished training.')
    self.model.save_params(self.args.save_models_dir + \
                           self.args.run_name + "/" + 'final.pth')
    self.model.save_params(self.args.save_models_dir + \
                           self.args.run_name + "/" + f'round_{round_num+1}.pth')
    self.monitor.maybe_eval_samplelog(self.model, round_num, allXtoR)
    
    print('Extrapolation')
    if self.args.temp_cond:
      query_temp_values = list(map(int, self.args.query_values.split(",")))
      for query_temp in query_temp_values:
          temp = torch.FloatTensor([query_temp]).repeat(self.args.query_num_samples).to(self.args.device).reshape(-1, 1)
          query_samples = self.model.batch_fwd_sample(self.args.query_num_samples,
                                                      temp=temp,
                                                      epsilon=0.0)
          with open(self.args.save_models_dir + \
                    self.args.run_name + "/" + f'query_samples_{query_temp}.pkl', 'wb') as f:
            pickle.dump(query_samples, f)
    else:
      query_samples = self.model.batch_fwd_sample(monitor_num_samples,
                                                  epsilon=0.0)
      with open(self.args.save_models_dir + \
                self.args.run_name + "/" + f'query_samples_{self.args.target_beta}.pkl', 'wb') as f:
        pickle.dump(query_samples, f)
    return

  def mode_seeking(self):
    allXtoR = dict()
    
    num_online = self.args.num_online_batches_per_round
    num_offline = self.args.num_offline_batches_per_round
    online_bsize = self.args.num_samples_per_online_batch
    offline_bsize = self.args.num_samples_per_offline_batch
    monitor_fast_every = self.args.monitor_fast_every
    monitor_num_samples = self.args.monitor_num_samples
    
    if self.args.temp_cond:
      temp_sampler = TemperatureConditional(self.args)

    print(f'Starting active learning. \
            Each round: num_online={num_online}, num_offline={num_offline}')
    total_samples = []
    for round_num in tqdm(range(self.args.num_active_learning_rounds)):
      print(f'Starting learning round {round_num+1} / {self.args.num_active_learning_rounds} ...')
      
      for _ in range(num_online):
        if self.args.temp_cond:
          temp = temp_sampler.sample(online_bsize, exp=True, 
                                                   fraction=round_num / self.args.num_active_learning_rounds)
          with torch.no_grad():
            explore_data = self.model.batch_fwd_sample(online_bsize,
                                                       temp=temp["beta_enc"],
                                                       epsilon=self.args.explore_epsilon)
          for _ in range(self.args.num_steps_per_batch):
            self.model.train(explore_data, temp=temp, online=True)
        else:
          with torch.no_grad():
            explore_data = self.model.batch_fwd_sample(online_bsize,
                                                         epsilon=self.args.explore_epsilon)
          for _ in range(self.args.num_steps_per_batch):
            self.model.train(explore_data, online=True)
            
        # Save to full dataset
        for exp in explore_data:
          if exp.x not in allXtoR:
            allXtoR[exp.x] = exp.r              
        total_samples.extend(explore_data)
        
      for _ in range(num_offline):
        offline_xs = self.select_offline_xs(allXtoR, offline_bsize)
        if self.args.temp_cond:
          temp = temp_sampler.sample(offline_bsize, exp=False,
                                                   fraction=round_num / self.args.num_active_learning_rounds)
          offline_dataset = self.offline_PB_traj_sample(offline_xs, allXtoR, temp=temp["beta_enc"])
          for _ in range(self.args.num_steps_per_batch):
            self.model.train(offline_dataset, temp=temp, online=False)
        else:
          offline_dataset = self.offline_PB_traj_sample(offline_xs, allXtoR)
          for _ in range(self.args.num_steps_per_batch):
            self.model.train(offline_dataset, online=False)
            
      # monitor
      if round_num % monitor_fast_every == 0 and round_num > 0:
        if self.args.temp_cond:
          temp = temp_sampler.sample(monitor_num_samples, exp=True, 
                                                          fraction=round_num / self.args.num_active_learning_rounds)
          truepolicy_data = self.model.batch_fwd_sample(monitor_num_samples,
                                                        temp=temp["beta_enc"],
                                                        epsilon=0.0)
        else:
          truepolicy_data = self.model.batch_fwd_sample(monitor_num_samples,
                                                        epsilon=0.0)
        self.monitor.log_samples(round_num, truepolicy_data)
      self.monitor.maybe_eval_samplelog(self.model, round_num, allXtoR)
      
      # save    
      if round_num and round_num % self.args.save_every_x_active_rounds == 0:
        self.model.save_params(self.args.save_models_dir + \
                               self.args.run_name + "/" + f'round_{round_num}.pth')
        with open(self.args.save_models_dir + \
                  self.args.run_name + "/" + f"round_{round_num}_sample.pkl", "wb") as f:
          pickle.dump(total_samples, f)
          
    print('Finished training.')
    self.model.save_params(self.args.save_models_dir + \
                           self.args.run_name + "/" + 'final.pth')
    self.model.save_params(self.args.save_models_dir + \
                           self.args.run_name + "/" + f'round_{round_num+1}.pth')
    with open(self.args.save_models_dir + \
          self.args.run_name + "/" + f"final_sample.pkl", "wb") as f:
      pickle.dump(total_samples, f)
    with open(self.args.save_models_dir + \
          self.args.run_name + "/" + f"round_{round_num+1}_sample.pkl", "wb") as f:
      pickle.dump(total_samples, f)
    self.monitor.maybe_eval_samplelog(self.model, round_num, allXtoR)
    return
  
  """
    Offline training
  """
  def select_offline_xs(self, allXtoR, batch_size):
    if self.args.prt:
      return self.__biased_sample_xs(allXtoR, batch_size)
    else:
      return self.__random_sample_xs(allXtoR, batch_size)
    
  def __biased_sample_xs(self, allXtoR, batch_size):
    """ Select xs for offline training. Returns List of [State].
        Draws 50% from top 10% of rewards, and 50% from bottom 90%. 
    """
    if len(allXtoR) < 10:
      return []
    rewards = np.array(list(allXtoR.values()))
    threshold = np.percentile(rewards, 90)
    top_xs = [x for x, r in allXtoR.items() if r >= threshold]
    bottom_xs = [x for x, r in allXtoR.items() if r <= threshold]
    sampled_xs = random.choices(top_xs, k=batch_size // 2) + \
                 random.choices(bottom_xs, k=batch_size // 2)
    return sampled_xs

  def __random_sample_xs(self, allXtoR, batch_size):
    """ Select xs for offline training. Returns List of [State]. """
    return random.choices(list(allXtoR.keys()), k=batch_size)

  def offline_PB_traj_sample(self, offline_xs, allXtoR, temp=None):
    """ Sample trajectories for x using P_B, for offline training with TB.
        Returns List of [Experience].
    """
    offline_rs = [allXtoR[x] for x in offline_xs]
    
    print(f'Sampling trajectories from backward policy ...')
    with torch.no_grad():
      if self.args.temp_cond:
        offline_trajs = self.model.batch_back_sample(offline_xs, temp=temp)
      else:
        offline_trajs = self.model.batch_back_sample(offline_xs)
    
    offline_dataset = [
      Experience(traj=traj, x=x, r=r,
                 logr=torch.log(torch.tensor(r, dtype=torch.float32, device=self.args.device)))
      for traj, x, r in zip(offline_trajs, offline_xs, offline_rs)
    ]
    return offline_dataset
