import argparse
import os
import warnings
from sys import platform


if platform != "win32":
    os.system("export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libGLEW.so")


parser = argparse.ArgumentParser(description='Args for general configs for BCIO')

# General
parser.add_argument('--gpu', default='-1', help='GPU Number.')
parser.add_argument('--alpha', type=str, default='./dataset/maze/POLICY/alpha/', help='where alpha is located')
parser.add_argument('--encoder', type=str, default='vgg', help='Which encoder should it use')
parser.add_argument('--choice', type=str, default='explore', help='Which encoder should it use')

parser.add_argument('--debug', action='store_true')
parser.add_argument('--verbose', action='store_true', help='If train should create progress bars.')
parser.add_argument('--sudoer', action='store_true', help='If user want to execute ASLA workaround. Should be sudo!')
parser.add_argument('--early_stop', action='store_true', help='Whether to early stop or not')

# Domain
parser.add_argument('--domain', default='maze', type=str)
parser.add_argument('--env_name', default='Ant-v3', type=str)
parser.add_argument('--vector', action='store_true')
parser.add_argument('--maze_size', default=5, help='How big maze should be (3, 5, 10).')
parser.add_argument('--run_name', type=str, default=None, help='Name of the folder')

# Models
parser.add_argument('--pretrained', action='store_true', help='Whether or not models should be pretrained')

# IDM
parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
parser.add_argument('--data_path', type=str, default='./dataset/maze/IDM/maze10/')
parser.add_argument('--idm_epochs', type=int, default=100, help='IDM number of epochs')
parser.add_argument('--lr', type=float, default=5e-3, help='Learning rate')
parser.add_argument('--lr_decay_rate', type=float, default=0.99, help='Learning rate decay')

parser.add_argument('--no_hit', action='store_true', help='')
parser.add_argument('--reducted', action='store_true', help='reducted')
parser.add_argument('--augmented', action='store_true', help='Use augmented data or nor')

# Policy
parser.add_argument('--expert_path', type=str)
parser.add_argument('--idm_weights', type=str, default=None, help='IDM Checkpoint')
parser.add_argument('--policy_batch_size', type=int, default=128, help='Batch size')
parser.add_argument('--policy_epochs', type=int, default=100, help='IDM number of epochs')
parser.add_argument('--policy_lr', type=float, default=1e-2, help='Learning rate')
parser.add_argument('--policy_lr_decay_rate', type=float, default=0.99, help='Learning rate decay')

args = parser.parse_args()

print('Using args')
args_dict = vars(args)
for i, arg in enumerate(args_dict):
    if i == 0:
        print('\nGeneral:')
    elif i == 13:
        print('\nModels:')
    elif i == 14:
        print('\nIDM:')
    elif i == 22:
        print('\nPolicy:')

    print(f'  {arg}: {args_dict[arg]}')

warnings.filterwarnings("ignore")
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Workaround for ALSA lib pcm.c:8424:(snd_pcm_recover) underrun occurred
if args.sudoer is True:
    print('\n')
    os.system('sudo aconnect -x')
