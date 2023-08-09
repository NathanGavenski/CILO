clear && python train_cilo.py \
--gpu $1 \
--pretrained \
--encoder vector \
--env_name hopper \
--run_name Hopper$2 \
--data_path ./dataset/hopper/random_hopper.npz \
--expert_path ./dataset/hopper/Hopper-v3.npz \
--alpha ./dataset/hopper/ALPHA3/ \
--domain vector \
--choice default \
\
--lr 1e-3 \
--lr_decay_rate 1 \
--batch_size 512 \
--idm_epochs 1000 \
\
--policy_lr 1e-3 \
--policy_lr_decay_rate 1 \
--policy_batch_size 512 \
\
--verbose
