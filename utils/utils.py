import os
from os import listdir
from os.path import isfile, join

import numpy as np
import torch


from datasets.VectorDataset import get_idm_vector_dataset
from datasets.VectorDataset import get_policy_vector_dataset
from Models.IDM import IDM
from Models.Policy import Policy
from utils.enjoy import get_environment
from utils.enjoy import play
from utils.exceptions import CheckpointAlreadyExists


def save_idm_model(model, replace=False, name=None, folder=None):
    parent_folder = './checkpoint/idm'
    path = folder if folder is not None else parent_folder
    if os.path.exists(path) is False:
        os.mkdir(path)

    onlyfiles = [f for f in listdir(path) if isfile(join(path, f))]
    checkpoint_name = f'model_{str(len(onlyfiles) + 1)}.ckpt' if name is None else name

    if checkpoint_name in onlyfiles and replace is False:
        raise CheckpointAlreadyExists(onlyfiles, checkpoint_name)
    elif replace is True:
        torch.save(model.state_dict(), f'{path}/best_model.ckpt')
    elif replace is False:
        torch.save(model.state.dict(), f'{path}/{checkpoint_name}')


def save_policy_model(model, replace=False, name=None, folder=None):
    parent_folder = './checkpoint/policy'
    path = folder if folder is not None else parent_folder
    if os.path.exists(path) is False:
        os.mkdir(path)

    onlyfiles = [f for f in listdir(path) if isfile(join(path, f))]
    checkpoint_name = f'model_{str(len(onlyfiles) + 1)}.ckpt' if name is None else name

    if checkpoint_name in onlyfiles and replace is False:
        raise CheckpointAlreadyExists(onlyfiles, checkpoint_name)
    elif replace is True:
        torch.save(model.state_dict(), f'{path}/best_model.ckpt')
    elif replace is False:
        torch.save(model.state.dict(), f'{path}/{checkpoint_name}')


def load_policy_model(args, environment, device, folder=None):
    parent_folder = './checkpoint/policy'
    path = folder if folder is not None else parent_folder

    model = Policy(
        environment['action'],
        net=args.encoder,
        pretrained=args.pretrained,
        input=environment['input_size']
    )
    model.load_state_dict(torch.load(f'{path}/best_model.ckpt'))
    model = model.to(device)
    model.eval()
    return model


def load_idm_model(args, device, folder=None):
    parent_folder = './checkpoint/policy'
    path = f'{parent_folder}/{folder}' if folder is not None else parent_folder

    model = IDM(args.actions, pretrained=args.pretrained)
    model.load_state_dict(torch.load(f'{path}/best_model.ckpt'))
    model = model.to(device)
    model.eval()
    return model


def save_gif(gif_images, random, iteration):
    if os.path.exists('./gifs/') is False:
        os.makedirs('./gifs/')

    name = 'Random' if random else 'Sample'
    gif_images[0].save(f'./gifs/{name}_{iteration}.gif',
                       format='GIF',
                       append_images=gif_images[1:],
                       save_all=True,
                       duration=100,
                       loop=0)


def policy_infer(
    model,
    dataloader,
    device,
    domain,
    episodes=10,
    bar=None,
    verbose=False,
    dataset=False,
    alpha_location=None,
    tensorboard=None,
    choice=None,
    deviation=None,
):
    transforms = dataloader.dataset.transforms
    get_performance = dataloader.dataset.get_performance

    if model.training:
        model.eval()

    total_solved = 0
    reward_epoch = []
    performance_epoch = []

    env = get_environment(domain)
    state = env.reset()
    run = {
        'states': np.ndarray((0, *state.shape)),
        'next_states': np.ndarray((0, *state.shape)),
        'actions': np.ndarray((0, *env.action_space.shape)),
        'starts': []
    }
    for e in range(episodes):
        env = get_environment(domain)

        play_function = domain['enjoy']
        with torch.no_grad():
            total_reward, goal, traj = play_function(
                env=env,
                model=model,
                dataset=dataset,
                transforms=transforms,
                device=device,
                tensorboard=tensorboard,
            )
        total_solved += goal

        if goal:
            run['states'] = np.append(run['states'], traj['states'], axis=0)
            run['next_states'] = np.append(run['next_states'], traj['next_states'], axis=0)
            run['actions'] = np.append(run['actions'], traj['actions'], axis=0)

            start = np.zeros(traj['states'].shape[0])
            start[0] = 1
            run['starts'] += start.astype(bool).tolist()

        performance_epoch.append(get_performance(total_reward))
        reward_epoch.append(total_reward)

        if verbose:
            print(f'{e}/{episodes} - Total Reward: {total_reward}')

        if bar is not None:
            bar.next()

        env.close()
        del env

    if not os.path.exists(alpha_location):
        os.makedirs(alpha_location)

    np.savez(f'{alpha_location}{domain["name"]}', **run)
    return (np.mean(reward_epoch), np.std(reward_epoch)), np.mean(performance_epoch), total_solved / episodes, run


domain = {
    'vector': {
        'idm_dataset': get_idm_vector_dataset,
        'policy_dataset': get_policy_vector_dataset,
        'enjoy': play,
    }
}
