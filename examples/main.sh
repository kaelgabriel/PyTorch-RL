#!/bin/bash

REWARD_FUNCTION=Normal
# REWARD_FUNCTION=Reward_16

SEED=42
# CUDA=True
ENV_NAME=Exploding4
NUM_EPOCHS=4000
# env_reset_mode=Discretized_Uniform
env_reset_mode=False
# env_reset_mode=Gaussian


STATE=New_Double

MODEL_INTERVAL=10

log_interval=1

CLIP_EPSILON=0.2
MINBATCH=1025
lr=0.0003
OPTIM_EPOCHS=15
OPTIM_BATCHSIZE=256
ENT_COEF=0.01
# ENT_COEF=0


# MODEL_PATH=/home/gabriel/repos/PyTorch-RL/checkpoint/name_NANPROBLEM_clip_0.2_minbatch_1024_lr_0.0001_optepochs_15_optbatchs_256_init_False_seed_42/NANPROBLEM_ppo_itr_1249.p
# MODEL_PATH=/home/gabriel/repos/PyTorch-RL/checkpoint/name_Exploding4_state_New_Double_reward_New_Double_reset_False_clip_0.2_minbatch_1000_lr_0.0001_optepochs_15_optbatchs_256_init_False_seed_42/Exploding4_ppo_itr_49.p

export OMP_NUM_THREADS=1
python drone_ppo.py --env-name=${ENV_NAME} --env_reset_mode=${env_reset_mode} --seed=${SEED} \
--max-iter-num=${NUM_EPOCHS} --save-model-interval=${MODEL_INTERVAL} --log-interval=${log_interval} --clip-epsilon=${CLIP_EPSILON} \
--min-batch-size=${MINBATCH} --learning-rate=${lr} --optim-epochs=${OPTIM_EPOCHS} --optim-batch-size=${OPTIM_BATCHSIZE} \
--reward=${REWARD_FUNCTION} --state=${STATE} --two-losses=1 --obs-running-state=1 --gpu-index=0 \
--ent-coef=${ENT_COEF} --log-prob-head=0 \
--save_path=name_${ENV_NAME}_state_${STATE}_reward_${STATE}_reset_${env_reset_mode}_clip_${CLIP_EPSILON}_minbatch_${MINBATCH}_lr_${lr}_optepochs_${OPTIM_EPOCHS}_optbatchs_${OPTIM_BATCHSIZE}_init_${env_reset_mode}_seed_${SEED}
# --ent-coef=${ENT_COEF}  --model-path=${MODEL_PATH} \



