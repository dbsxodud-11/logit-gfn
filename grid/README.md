# Flow Network based Generative Models for Non-Iterative Diverse Candidate Generation

Implementation for [our paper](https://arxiv.org/abs/2106.04399), submitted to NeurIPS 2021 (also check this high-level [blog post](http://folinoid.com/w/gflownet)).

This is a minimum working version of the code used for the paper, which is extracted from the internal repository of the [Mila Molecule Discovery](https://mila.quebec/en/ai-society/exascale-search-of-molecules/) project. Original commits are lost here, but the credit for this code goes to [@bengioe](https://github.com/bengioe), [@MJ10](https://github.com/MJ10) and [@MKorablyov](https://github.com/MKorablyov/) (see paper).

Note: for more modern implementations of GFlowNet, check out [recursionpharma/gflownet](https://github.com/recursionpharma/gflownet), [saleml/gfn](https://github.com/saleml/gfn), and [alexhernandezgarcia/gflownet](https://github.com/alexhernandezgarcia/gflownet/tree/continuous).

## Grid experiments

Requirements for base experiments: 
- `torch numpy scipy tqdm`


## Running Experiments

For TB, you can run the following command:
```
python trainer.py --ndim 4 --horizon 16 --n_train_steps 5000 --eval_interval 500 --agent tb --logger wandb
```

For Temperature-conditional GFlowNets, you can run the following command:
```
python trainer.py --ndim 4 --horizon 16 --n_train_steps 5000 --eval_interval 500 --agent condtb --cond_type logit+layer --logger wandb

python trainer.py --ndim 4 --horizon 16 --n_train_steps 5000 --eval_interval 500 --agent condtb --cond_type logit --logger wandb

python trainer.py --ndim 4 --horizon 16 --n_train_steps 5000 --eval_interval 500 --agent condtb --cond_type layer --logger wandb
```
