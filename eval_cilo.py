from utils.args import args
from utils.utils import domain
from utils.enjoy import get_environment
from Models.Policy import Policy
import torch
import numpy as np
from tqdm import tqdm


env_info = {
    "cheetah": {
        "expert": 7561.78,
        "random": -293.13
    },
    "hopper": {
        "expert": 3589.88,
        "random": 17.92
    },
    "swimmer": {
        "expert": 259.52,
        "random": 0.73
    },
    "pendulum": {
        "expert": 1000,
        "random": 5.70
    },
    "ant": {
        "expert": 5544.65,
        "random": -65.11
    },
}


def normalise(agent, expert, random):
    return (agent - random) / (expert - random)


def get_performance(reward, name):
    return normalise(reward, env_info[name]["expert"], env_info[name]["random"])


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    environment = domain["vector"]
    environment["name"] = args.env_name

    env = get_environment(environment)
    action_dimension = env.action_space.shape[0]
    state_size = env.reset().shape[0]
    policy_model = Policy(action_dimension, input=state_size)
    policy_model.to(device)
    policy_model.load_state_dict(torch.load(f"{args.pretrain_path}best_policy.pt"))
    policy_model.eval()
    print()

    with torch.no_grad():
        acc_epoch_reward = []
        epoch_performance = []

        pbar = tqdm(range(50))
        for _ in pbar:
            acc_reward = 0
            done = False
            obs = env.reset()
            while not done:
                obs = torch.tensor(obs)[None]
                action = policy_model(obs)[0]
                action = torch.clip(action, -1, 1)
                obs, reward, done, info = env.step(action.numpy())
                acc_reward += reward
            acc_epoch_reward.append(acc_reward)
            epoch_performance.append(get_performance(acc_reward, args.env_name))
            pbar.set_description_str(f"AER: {np.mean(acc_epoch_reward)}")

    print()
    print(np.mean(acc_epoch_reward), np.std(acc_epoch_reward))
    print(np.mean(epoch_performance), np.std(epoch_performance))
    print()

    acc_epoch_reward.sort(reverse=True)
    epoch_performance.sort(reverse=True)

    acc_epoch_reward = acc_epoch_reward[:10]
    epoch_performance = epoch_performance[:10]
    print()
    print(np.mean(acc_epoch_reward), np.std(acc_epoch_reward))
    print(np.mean(epoch_performance), np.std(epoch_performance))
    print()
