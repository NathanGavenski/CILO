clear && python train_cilo.py \
--gpu $1 \
--pretrained \
--encoder vector \
--env_name walker \
--run_name Walker$2 \
--data_path ./dataset/walker/random_walker.npz \
--expert_path ./dataset/walker/Walker2d-v2.npz \
--alpha ./dataset/walker/ALPHA2/ \
--domain vector \
--choice explore \
\
--lr 7e-4 \
--lr_decay_rate 1 \
--batch_size 1024 \
--idm_epochs 1000 \
\
--policy_lr 1e-3 \
--policy_lr_decay_rate 1 \
--policy_batch_size 1024 \
\
--verbose
