#!/bin/bash
# FILE=/home/gabriel/repos/PyTorch-RL/checkpoint/name_Testing_norm_deterministic_clip_0.2_minbatch_2048_lr_0.0003_optepochs_10_optbatchs_64_init_Discretized_Uniform_seed_42_GAIL/GAIL_Testing_norm_deterministic_itr_799.p
#/home/gabriel/repos/PyTorch-RL/checkpoint/name_Testing_norm_stochastic_clip_0.2_minbatch_2048_lr_0.0003_optepochs_10_optbatchs_64_init_Discretized_Uniform_seed_42_GAIL/GAIL_Testing_norm_stochastic_itr_99.p 

FILE=/home/gabriel/repos/PyTorch-RL/assets/learned_models/Teste_ppo_499.p
# FILE=/home/gabriel/repos/PyTorch-RL/checkpoint/name_Testing_norm_deterministic_clip_0.2_minbatch_2048_lr_0.0003_optepochs_10_optbatchs_64_init_Discretized_Uniform_seed_42_GAIL/GAIL_Testing_norm_deterministic_itr_349.p


python evaluate_drone.py --file=${FILE} --env_reset_mode=Discretized_Uniform --max_timesteps=3000 --headless


