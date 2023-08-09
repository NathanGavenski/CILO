clear && python train_cilo.py \
--gpu $1 \
--pretrained \
--encoder vector \
--env_name cheetah \
--run_name Cheetah$2 \
--data_path ./dataset/cheetah/random_cheetah.npz \
--expert_path ./dataset/cheetah/HalfCheetah-v2.npz \
--alpha ./dataset/cheetah/ALPHA/ \
--domain vector \
--choice explore \
\
--lr 1e-3 \
--lr_decay_rate 1 \
--batch_size 1024 \
--idm_epochs 1000 \
\
--policy_lr 1e-3 \
--policy_lr_decay_rate 1 \
--policy_batch_size 1024 \
\
--verbose
