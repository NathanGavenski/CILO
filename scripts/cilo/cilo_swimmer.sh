clear && python train_cilo.py \
--gpu $1 \
--pretrained \
--encoder vector \
--env_name swimmer \
--run_name Swimmer$2 \
--data_path ./dataset/swimmer/random_swimmer.npz \
--expert_path ./dataset/swimmer/Swimmer-v2.npz \
--alpha ./dataset/swimmer/ALPHA/ \
--domain vector \
--choice explore \
\
--lr 3e-3 \
--lr_decay_rate 1 \
--batch_size 1024 \
--idm_epochs 1000 \
\
--policy_lr 7e-4 \
--policy_lr_decay_rate 1 \
--policy_batch_size 1024 \
\
--verbose
