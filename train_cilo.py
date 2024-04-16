# Args should be imported before everything to cover https://discuss.pytorch.org/t/cuda-visible-device-is-of-no-use/10018
import mujoco_py
from utils.args import args
import numpy as np
import os
import torch

from progress.bar import Bar
from torch import nn, optim

from Models.General.MLP import MLP
from Models.IDM import IDM
from Models.IDM import train as train_idm
from Models.IDM import validation as validate_idm
from Models.Policy import Policy
from Models.Policy import train as train_policy
from Models.Policy import validation as validate_policy
from utils.board import Board
from utils.utils import domain
from utils.utils import policy_infer
from utils.enjoy import get_environment
from utils.adversarial import adversarial


################ ARGS: GPU and Pretrained ################
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
environment = domain[args.domain]
if args.domain == 'vector':
    environment['name'] = args.env_name

################ Tensorboard ################
parent_folder = "_".join(args.run_name.split('_')[:-1])
st_char = environment['name'][0].upper()
rest_char = environment['name'][1:]
env_name = f'{st_char}{rest_char}'

name = f'./checkpoint/alpha/{env_name}/{parent_folder}/{args.run_name}'
if os.path.exists(name) is False:
    os.makedirs(name)

parent_folder = "_".join(args.run_name.split('_')[:-1])
path = f'./runs/alpha/{env_name}/{parent_folder}/{args.run_name}'
if os.path.exists(path) is False:
    os.makedirs(path)

board = Board(name, path)

################ Datasets ################
print('\nCreating PyTorch IDM Datasets')
print(f'Using dataset: {args.data_path} with batch size: {args.batch_size}')
get_idm_dataset = environment['idm_dataset']
idm_train, idm_validation = get_idm_dataset(
    args.data_path,
    args.batch_size
)
print(f'IDM Train dataset length: {idm_train.dataset.states.shape[0]}')
print(f'IDM Eval dataset length: {idm_validation.dataset.states.shape[0]}')

print('\nCreating PyTorch Policy Datasets')
print(f'Using dataset: {args.expert_path} with batch size: {args.policy_batch_size}')
get_policy_dataset = environment['policy_dataset']
policy_train, policy_validation = get_policy_dataset(
    args.expert_path,
    args.data_path,
    args.policy_batch_size,
)
policy_eval_length = len(policy_validation) if isinstance(policy_validation, list) else policy_validation.dataset.states.shape[0]
print(f'Expert Train dataset length: {policy_train.dataset.states.shape[0]}')
print(f'Expert Eval dataset length: {policy_eval_length}')
print(f'Expert reward: {policy_train.dataset.expert} Random reward: {policy_train.dataset.random}')

################ Model and action size ################
print('\nCreating Models')
env = get_environment(environment)
action_dimension = env.action_space.shape[0]
state_size = env.reset().shape[0]
policy_model = Policy(action_dimension, input=state_size)
idm_model = IDM(action_dimension, input=state_size * 2)
discriminator_model = MLP(policy_train.dataset.signatures.size(-1), 2)

policy_model.to(device)
idm_model.to(device)
discriminator_model.to(device)

################ Optimizer and loss ################
print('\nCreating Optimizer and Loss')
print(f'IDM learning rate: {args.lr}\nPolicy learning rate: {args.policy_lr}')
idm_lr = args.lr
idm_criterion = nn.L1Loss()
idm_optimizer = optim.Adam(idm_model.parameters(), lr=idm_lr) #, weight_decay=1e-2)

policy_lr = args.policy_lr
policy_criterion = nn.L1Loss()
policy_optimizer = optim.Adam(policy_model.parameters(), lr=policy_lr) # , weight_decay=1e-2)

disc_lr = 5e-3
disc_criterion = nn.CrossEntropyLoss()
disc_optimizer = optim.Adam(discriminator_model.parameters(), lr=disc_lr)

################ Learning rate decay ################
print('Setting up Learning Rate Decay function and Schedulers')

idm_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer=idm_optimizer,
    factor=0.9,
    threshold_mode='abs',
    patience=10,
)

policy_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer=policy_optimizer,
    factor=0.9,
    threshold_mode='abs'
)

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

################ Train ################
print('Starting Train\n')
best_epoch_perf = 0
early_stop_count = 0
max_epochs = args.idm_epochs
max_iter = len(idm_train) + len(idm_validation) + len(policy_train) + len(policy_validation)

for epoch in range(max_epochs):

    if epoch > (max_epochs * 0.9):
        args.choice = "greedy"

    # IDM Train
    if args.verbose is True:
        bar = Bar(f'EPOCH {epoch:3d}', max=max_iter, suffix='%(percent).1f%% - %(eta)ds')

    batch_loss = []
    batch_acc = torch.Tensor(size=(0, action_dimension))
    for itr, mini_batch in enumerate(idm_train):
        loss, acc = train_idm(idm_model, mini_batch, idm_criterion, idm_optimizer, device)

        batch_acc = torch.cat((batch_acc, acc[None].cpu()), dim=0)
        batch_loss.append(loss)

        if args.verbose is True:
            bar.next()

        if args.debug:
            break

    for i, acc in enumerate(batch_acc.mean(dim=0)):
        board.add_scalars(
            train=True,
            **{f'IDM_Acc_action_{i}': acc},
        )

    board.add_scalars(
        train=True,
        IDM_Loss=np.array(batch_loss).mean()
    )
    board.add_scalar('IDM Learning Rate', get_lr(idm_optimizer))

    exploration = batch_acc.mean(dim=0)

    # IDM Validation
    #batch_loss = []
    batch_acc = torch.Tensor(size=(0, action_dimension))
    for itr, mini_batch in enumerate(idm_validation):
        with torch.no_grad():
            _, acc = validate_idm(idm_model, mini_batch, device, board)
        batch_acc = torch.cat((batch_acc, acc[None].cpu()), dim=0)
        #batch_loss.append(loss)

        if args.verbose is True:
            bar.next()

        if args.debug:
            break

    for i, acc in enumerate(batch_acc.mean(dim=0)):
        board.add_scalars(
            train=False,
            **{f'IDM_Acc_action_{i}': acc},
        )
    # idm_scheduler.step(np.mean(batch_loss))

    # Policy Train
    exploration = batch_acc.mean(dim=0)
    batch_acc = torch.Tensor(size=(0, action_dimension))
    batch_idm_acc = torch.Tensor(size=(0, action_dimension))
    batch_loss = []
    for itr, mini_batch in enumerate(policy_train):
        loss, acc, idm_acc = train_policy(
            policy_model,
            idm_model,
            mini_batch,
            policy_criterion,
            policy_optimizer,
            device,
            args,
            tensorboard=board,
            actions=exploration
        )

        batch_acc = torch.cat((batch_acc, acc[None].cpu()), dim=0)
        batch_idm_acc = torch.cat((batch_idm_acc, idm_acc[None].cpu()), dim=0)
        batch_loss.append(loss)

        if args.verbose is True:
            bar.next()

        if args.debug:
            break

    for i, (acc, idm_acc) in enumerate(zip(batch_acc.mean(dim=0), batch_idm_acc.mean(dim=0))):
        board.add_scalars(
            train=True,
            **{f'Policy_Acc_action_{i}': acc, f'IDM_GT_Error_Action_{i}': idm_acc},
        )

    board.add_scalars(
        train=True,
        Policy_Loss=np.mean(batch_loss),
    )
    board.add_scalar('Policy Learning Rate', get_lr(policy_optimizer))

    # Policy Validation
    batch_acc = torch.Tensor(size=(0, action_dimension))
    for itr, mini_batch in enumerate(policy_validation):
        acc = validate_policy(
            policy_model,
            idm_model,
            mini_batch,
            device,
            args,
            actions=exploration,
            tensorboard=board,
        )

        batch_acc = torch.cat((batch_acc, acc[None].cpu()), dim=0)
        bar.next()

    for i, acc in enumerate(batch_acc.mean(dim=0)):
        board.add_scalars(
            train=False,
            **{f'Policy_Acc_action_{i}': acc}
        )

    # Policy Eval
    if args.debug:
        amount = 1
    else:
        amount = 50

    if args.verbose is True:
        bar = Bar(f'VALID Sample {epoch:3d}', max=amount, suffix='%(percent).1f%% - %(eta)ds')
    else:
        bar = None

    (infer, std), performance, solved, run = policy_infer(
        policy_model,
        dataloader=policy_train,
        device=device,
        domain=environment,
        bar=bar,
        episodes=amount,
        alpha_location=args.alpha,
        dataset=True,
        tensorboard=board,
        choice=args.choice,
    )

    board.add_scalars(
        train=False,
        AER_Sample=infer,
        AER_std=std,
        Sample_Solved=solved,
        Performance_Sample=performance
    )

    if best_epoch_perf < performance:
        best_epoch_perf = performance
        path = f'/checkpoint/{environment["name"]}/'
        if not os.path.exists(path):
            os.makedirs(path)

        torch.save(idm_model.state_dict(), f'{path}{epoch}_idm.pt')
        torch.save(policy_model.state_dict(), f'{path}{epoch}_policy.pt')
        torch.save(discriminator_model.state_dict(), f'{path}{epoch}_disc.pt')

    print(f'\nSample Infer - AER: {round(infer, 4)} Performance: {round(performance, 4)}')

    run, solved, _ = adversarial(
        policy_train.dataset,
        run,
        disc_optimizer,
        disc_criterion,
        discriminator_model,
        4,
        device,
        board
    )

    print(f'Discriminator information - Fooled trajectories: {solved * 100}%')
    print('Getting IDM dataset\n')

    board.add_scalars(
        train=True,
        IDM_Size=idm_train.dataset.states.shape[0]
    )
    board.add_scalars(
        train=False,
        IDM_Size=idm_validation.dataset.states.shape[0]
    )

    # solved -> percentage [0, 1]
    i_pre = 1 - solved
    i_pos = solved

    alpha_proportion = int(run['states'].shape[0] * i_pre)
    if alpha_proportion > 0:
        train_proportion = int(idm_train.dataset.states.shape[0] * i_pre)
        idm_train.dataset.states = np.append(
            idm_train.dataset.states,
            run['states'][:alpha_proportion],
            axis=0
        )

        eval_proportion = int(idm_validation.dataset.states.shape[0] * i_pre)
        idm_validation.dataset.states = np.append(
            idm_validation.dataset.states,
            run['states'][alpha_proportion:],
            axis=0
        )

    alpha_proportion = int(run['next_states'].shape[0] * i_pre)
    if alpha_proportion > 0:
        train_proportion = int(idm_train.dataset.next_states.shape[0] * i_pre)
        idm_train.dataset.next_states = np.append(
            idm_train.dataset.next_states,
            run['next_states'][:alpha_proportion],
            axis=0
        )

        eval_proportion = int(idm_validation.dataset.next_states.shape[0] * i_pre)
        idm_validation.dataset.next_states = np.append(
            idm_validation.dataset.next_states,
            run['next_states'][alpha_proportion:],
            axis=0
        )

    alpha_proportion = int(run['actions'].shape[0] * i_pre)
    if alpha_proportion > 0:
        train_proportion = int(idm_train.dataset.actions.shape[0] * i_pre)
        idm_train.dataset.actions = np.append(
            idm_train.dataset.actions,
            run['actions'][:alpha_proportion],
            axis=0
        )

        eval_proportion = int(idm_validation.dataset.actions.shape[0] * i_pre)
        idm_validation.dataset.actions = np.append(
            idm_validation.dataset.actions,
            run['actions'][alpha_proportion:],
            axis=0
        )

    # Necessary updates
    board.advance()
