import torch
from utils.math import *

from torch.distributions import MultivariateNormal, Normal

# def ppo_step(policy_net, value_net, optimizer_policy, optimizer_value, optim_value_iternum, states, actions,
#              returns, advantages, fixed_log_probs, clip_epsilon, l2_reg, scheduler_policy, scheduler_value,
#              ent_coef = 0, vf_coef=0.5):

def ppo_step(policy_net, value_net, unique_optimizer, optim_value_iternum, states, actions,
             returns, advantages, fixed_log_probs, clip_epsilon, l2_reg, scheduler,
             ent_coef = 0, vf_coef=0.5):

    # advantages = returns - values
    values = returns - advantages

    # Normalize the advantages
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    OLDVPRED = values
    """update critic"""
    for _ in range(optim_value_iternum):
        values_pred = value_net(states)
        #vpredclipped = OLDVPRED + tf.clip_by_value(train_model.vf - OLDVPRED, - CLIPRANGE, CLIPRANGE)
        vpredclipped = OLDVPRED + torch.clamp(values_pred - OLDVPRED,min =  - clip_epsilon, max =  clip_epsilon)
        # Unclipped value
        vf_losses1 = (values_pred - returns).pow(2)
        # Clipped value
        vf_losses2 = (vpredclipped - returns).pow(2)
        
        value_loss = .5 * torch.max(vf_losses1, vf_losses2).mean()
        # value_loss = (values_pred - returns).pow(2).mean()
        # weight decay
        for param in value_net.parameters():
            value_loss += param.pow(2).sum() * l2_reg
        # optimizer_value.zero_grad()
        # value_loss.backward()
        # torch.nn.utils.clip_grad_norm_(value_net.parameters(), 0.5)

        # optimizer_value.step()
        # # Update LR
        # scheduler_value.step()
    """update policy"""
    log_probs , action_mean, action_std = policy_net.get_log_prob(states, actions)
    # Calculate the entropy
    dist = Normal(action_mean, action_std)
    entropy = dist.entropy().mean()

    # Calculate the explained_variance
    # print(values_pred.squeeze().shape,returns.squeeze().shape)
    try:
        ev = explained_variance(values_pred.squeeze(),returns.squeeze())
    except:
        ev=np.nan


    ratio = torch.exp(log_probs - fixed_log_probs)
    ## Calculate clipfrac
    #clipfrac = tf.reduce_mean(tf.to_float(tf.greater(tf.abs(ratio - 1.0), CLIPRANGE)))
    # clipfrac = tf.reduce_mean(tf.greater(tf.abs(ratio - 1.0), CLIPRANGE)))
    clipfrac =  (torch.gt(torch.abs(ratio - 1), clip_epsilon)).float().mean().item()
    ## Calculate Approx KL
    # approxkl = .5 * tf.reduce_mean(tf.square(neglogpac - OLDNEGLOGPAC))
    diff = log_probs - fixed_log_probs
    approxkl = .5 * torch.mean(torch.mul(diff,diff))
    
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * advantages
    # policy_surr = -torch.min(surr1, surr2).mean()
    policy_loss = -torch.max(surr1, surr2).mean()


    # Total loss
    loss = policy_loss - entropy * ent_coef + value_loss * vf_coef
    #loss.backward
    # print('loss = ', loss)
    

    #unique optimizer 
    # params = list(policy_net.parameters()) + list(value_net.parameters())
    unique_optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(policy_net.parameters(), 0.5)
    torch.nn.utils.clip_grad_norm_(value_net.parameters(), 0.5)
    unique_optimizer.step()



    scheduler.step()

    return policy_loss.item(), value_loss.item(), ev, clipfrac, entropy.item(), approxkl.item()