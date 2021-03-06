#!/bin/bash

# REWARD_FUNCTION=Normal

SEED=42
# CUDA=True
ENV_NAME=Optmizations_PPO
NUM_EPOCHS=2000
env_reset_mode=Discretized_Uniform

MODEL_INTERVAL=50

log_interval=1

CLIP_EPSILON=0.2
MINBATCH=2048
lr=0.0003
OPTIM_EPOCHS=10
OPTIM_BATCHSIZE=64



export OMP_NUM_THREADS=1
python drone_ppo.py --env-name=${ENV_NAME} --env_reset_mode=${env_reset_mode} --seed=${SEED} \
--max-iter-num=${NUM_EPOCHS} --save-model-interval=${MODEL_INTERVAL} --log-interval=${log_interval} --clip-epsilon=${CLIP_EPSILON} \
--min-batch-size=${MINBATCH} --learning-rate=${lr} --optim-epochs=${OPTIM_EPOCHS} --optim-batch-size=${OPTIM_BATCHSIZE} \
--save_path=name_${ENV_NAME}_clip_${CLIP_EPSILON}_minbatch_${MINBATCH}_lr_${lr}_optepochs_${OPTIM_EPOCHS}_optbatchs_${OPTIM_BATCHSIZE}_init_${env_reset_mode}_seed_${SEED} \
--sac-policy


