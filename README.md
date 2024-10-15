# Continuous Imitation Learning from Observation (CILO)
Pytorch official implementation for Continuous Imitation Learning from Observation from [Explorative Imitation Learning: A Path Signature Approach for Continuous Environments](https://kclpure.kcl.ac.uk/portal/en/publications/explorative-imitation-learning-a-path-signature-approach-for-cont) (ECAI).

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

### Using samples from paper:
If you want to use the same datasets from the paper they are all publicly available via [IL-Datasets](https://github.com/NathanGavenski/IL-Datasets).
All datasets are listed on [HuggingFace](https://huggingface.co/collections/NathanGavenski/cilo-datasets-670e862c84a8c371ccb6ce2d) and can be downloaded using BaselineDataset from IL-Datasets.
To use the dataset:

```python
from imitation_datasets.dataset import BaselineDataset

dataset = BaselineDataset("NathanGavenski/Ant-v2", source="huggingface")
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

### Ciation
```latex
@incollection{gavenski2024explorative,
	title={Explorative Imitation Learning: A Path Signature Approach for Continuous Environments},
	author={Gavenski, Nathan and Monteiro, Juarez and Meneguzzi, Felipe and Luck, Michael and Rodrigues, Odinaldo},
	booktitle={ECAI 2024},
	pages={}
	year={2024},
	publisher={IOS Press}
}
```
