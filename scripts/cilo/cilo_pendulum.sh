clear && python train_cilo.py \
--gpu $1 \
--pretrained \
--encoder vector \
--env_name pendulum \
--run_name Pendulum$2 \
--data_path ./dataset/pendulum/random_pendulum.npz \
--expert_path ./dataset/pendulum/InvertedPendulum-v2.npz \
--alpha ./dataset/pendulum/ALPHA/ \
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
