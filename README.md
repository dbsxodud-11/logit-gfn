# Logit-GFN
Official Code for Learning to Scale Logits for Temperature-conditional GFlowNets

### Environment Setup
To install dependecies, please run the command `pip install -r requirement.txt`.
Note that python version should be < 3.8 for running RNA-Binding tasks. You should install `pyg` with the following command
```
pip install torch_geometric
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-1.13.0+cu117.html
```

### Code references
Our implementation is heavily based on "Towards Understanding and Improving GFlowNet Training" (https://github.com/maxwshen/gflownet). 

### Large Files

You can download additional large files by following link:
These files should be placed in `datasets`

### Offline Generalization
You can run the following command to valiate the effectiveness of Logit-GFN on offline generalization. As a default setting, Logit-GFN and Layer-GFN are trained with $\beta\sim\text{Unif}[10, 50]$ and evaluate with multiple $\beta$ values from $\beta=1$ to $\beta=5,000$.
```
# Logit-GFN
python main.py --task offline_generalization --setting tfbind8 --temp_cond --temp_cond_type logit

# Layer-GFN
python main.py --task offline_generalization --setting tfbind8 --temp_cond --temp_cond_type layer

# Unconditional GFN
python main.py --task offline_generalization --setting tfbind8 --target_beta 5000
```

### Online Mode Seeking
You can run the following command to validate the effectiveness of Logit-GFN on online mode seeking problems. As a default setting, Logit-GFN and Layer-GFN are trained with $\beta\sim\text{Unif}[1, 3]$ and explore with $\beta\sim\text{Unif}[1, 3]$. For unconditional GFN, we set $\beta=1$ as a default setting.
```
# Logit-GFN
python main.py --task mode_seeking.py --setting tfbind8 --temp_cond --temp_cond_type logit

# Layer-GFN
python main.py --task mode_seeking.py --setting tfbind8 --temp_cond --temp_cond_type layer

# Unconditional GFN
python main.py --task mode_seeking.py --setting tfbind8 --target_beta 1
```

### Additional Experiments
You can change various biochemical tasks to evaluate the performance of Logit-GFN by setting `--setting` option.
- Available Options: `qm9str, sehstr, tfbind8, rna`
```
python main.py --task mode_seeking.py --setting <setting> --temp_cond --temp_cond_type logit
```

You can change GFlowNet training objectives to evaluate the performance of Logit-GFN by setting `--loss_type` option.
- Available Options: `tb, maxent, db, subtb`
```
python main.py --task mode_seeking.py --setting tfbind8 --temp_cond --temp_cond_type logit --loss_type <loss_type>
```

You can turn on layer-conditioning option for Logit-GFN by setting `--layer-conditioning`.
```
python main.py --task mode_seeking.py --setting tfbind8 --temp_cond --temp_cond_type logit --layer-conditioning
```

You can turn on thermometer embedding option for Layer-GFN by setting `--thermometer`.
```
python main.py --task mode_seeking.py --setting tfbind8 --temp_cond --temp_cond_type logit --thermometer
```

You can adjust the number of gradient steps per batch ($K$) by setting `--num_steps_per_batch`.
```
python main.py --task mode_seeking.py --setting tfbind8 --temp_cond --temp_cond_type logit --num_steps_per_batch <K>
```

You can change $P_{\text{train}}(\beta)$, temperatue sampling distribution for Logit-GFN by setting `--train_temp_dist` option. There are various suboptions to decide depending on the distribution.
```
# Constant
python main.py --task mode_seeking.py --setting tfbind8 --temp_cond --temp_cond_type logit --train_temp_dist constant --train_temp 1

# Uniform
python main.py --task mode_seeking.py --setting tfbind8 --temp_cond --temp_cond_type logit --train_temp_dist uniform --train_temp_min 1 --train_temp_max 3

# LogUniform
python main.py --task mode_seeking.py --setting tfbind8 --temp_cond --temp_cond_type logit --train_temp_dist loguniform --train_temp_min 1 --train_temp_max 3

# ExpUniform
python main.py --task mode_seeking.py --setting tfbind8 --temp_cond --temp_cond_type logit --train_temp_dist expuniform --train_temp_min 1 --train_temp_max 3

# Normal
python main.py --task mode_seeking.py --setting tfbind8 --temp_cond --temp_cond_type logit --train_temp_dist normal --train_temp_min 1 --train_temp_max 3 --train_temp_mu 2 --train_temp_sigma 0.5

# Simulated Annealing
python main.py --task mode_seeking.py --setting tfbind8 --temp_cond --temp_cond_type logit --train_temp_dist annealing --train_temp_min 1 --train_temp_max 3

# Simulated Annealing (Inverse)
python main.py --task mode_seeking.py --setting tfbind8 --temp_cond --temp_cond_type logit --train_temp_dist annealing-inv --train_temp_min 1 --train_temp_max 3
```

You can also change $P_{\text{exp}}(\beta)$, temperatue sampling distribution for Logit-GFN by setting `--exp_temp_dist` option. There are various suboptions to decide depending on the distribution.
```
# Constant
python mode_seeking.py --task tfbind8 --temp_cond --temp_cond_type logit --exp_temp_dist constant --exp_temp 1

# Uniform
python mode_seeking.py --task tfbind8 --temp_cond --temp_cond_type logit --exp_temp_dist uniform --exp_temp_min 1 --exp_temp_max 3

# LogUniform
python mode_seeking.py --task tfbind8 --temp_cond --temp_cond_type logit --exp_temp_dist loguniform --exp_temp_min 1 --exp_temp_max 3

# ExpUniform
python mode_seeking.py --task tfbind8 --temp_cond --temp_cond_type logit --exp_temp_dist expuniform --exp_temp_min 1 --exp_temp_max 3

# Normal
python mode_seeking.py --task tfbind8 --temp_cond --temp_cond_type logit --exp_temp_dist normal --exp_temp_min 1 --exp_temp_max 3 --exp_temp_mu 2 --exp_temp_sigma 0.5

# Simulated Annealing
python mode_seeking.py --task tfbind8 --temp_cond --temp_cond_type logit --exp_temp_dist annealing --exp_temp_min 1 --exp_temp_max 3

# Simulated Annealing (Inverse)
python mode_seeking.py --task tfbind8 --temp_cond --temp_cond_type logit --exp_temp_dist annealing-inv --exp_temp_min 1 --exp_temp_max 3
```
