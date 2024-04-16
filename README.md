# Continuous Imitation Learning from Observation (CILO)
Pytorch official implementation for Continuous Imitation Learning from Observation

## Requirements
```
Python: 3.9.15
Conda: 23.1.0
```

### Installing dependencies
There is a script in `./dependencies/install.sh`, which will create a conda environment and install all dependencies needed to run this repository.

## Running 

To run CILO you need to first create random transition samples ($I^{pre}$) and expert samples ($\mathcal{T}^e$).

### Creating random samples:

To create random samples for **one** environment:
```
python create_random_mujoco.py --env_name <ENV> --data_path <PATH>
```

For example:
```
python create_random_mujoco.py --env_name Ant-v3 --data_path ./dataset/ant/random_ant
```

To create random samples for **all** environments:
```
bash ./scripts/create_randoms.sh
```

### Creating expert samples:

To create expert samples for **one** environment:
```
python create_dataset_mujoco.py -t <THREADS> -e <EPISODES> -g <ENV> --mode <play|collate|all>
```

For example:
```
python create_dataset_mujoco.py -t 4 -e 10 -g ant --mode all
```

To create expert samples for **all** environments:
```
bash ./scripts/create_experts.sh
```


### Running CILO

To run CILO you can run the command:
```
clear && python train_cilo.py \
--gpu <GPU> \
--encoder vector \
--env_name <ENV> \
--run_name <RUN NAME> \
--data_path <RANDOM> \
--expert_path <EXPERT> \
--alpha <ALPHA> \
--domain vector \
--choice explore \
\
--lr <Dynamics LR> \
--lr_decay_rate <LR DECAY> \
--batch_size <BATCH SIZE> \
--idm_epochs <EPOCHS> \
\
--policy_lr <Policy LR> \
--policy_lr_decay_rate <LR DECAY> \
--policy_batch_size <BATCH SIZE> \
\
--verbose
```
where `<GPU>` should be `-1` if there are no GPUs available, `<RANDOM>` is the path for the random samples, `<EXPERT>` is the path for the expert samples, and the `<RUN NAME>` is the name you want for you experiment in the tensorboard.

For simplicity, we provide a script for each environment with all hyperparameters used during training. To use them:
```
bash ./scripts/cilo/cilo_ant.sh -1 experiment1
```
where the first argument is the gpu number and the second is the experiment name.