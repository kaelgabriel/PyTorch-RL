import argparse
import gym
import os
import sys
import pickle
import time
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils import *
from models.mlp_policy import Policy
from models.mlp_critic import Value
from models.mlp_policy_disc import DiscretePolicy
from core.ppo import ppo_step_one_loss, ppo_step_two_losses
from core.common import estimate_advantages
from core.agent import Agent

from larocs_sim.envs.drone_env import DroneEnv
import csv

from models.mlp_sac_policy import Policy_Tanh_Gaussian


def terminate():
    try:
        env.shutdown();import sys; sys.exit(0)
    except:
        import sys; sys.exit(0)

def check_dir(file_name):
    directory = os.path.dirname(file_name)
    if not os.path.exists(directory):
        os.makedirs(directory)


parser = argparse.ArgumentParser(description='PyTorch PPO example')
parser.add_argument('--env-name', default="Hopper-v2", metavar='G',
                    help='name of the environment to run')
parser.add_argument('--env_reset_mode', default="Discretized_Uniform",
                    help='name of the environment to run')

parser.add_argument('--model-path', metavar='G',
                    help='path of pre-trained model')
parser.add_argument('--render', action='store_true', default=False,
                    help='render the environment')
parser.add_argument('--log-std', type=float, default=-0.0, metavar='G',
                    help='log std for the policy (default: -0.0)')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor (default: 0.99)')
parser.add_argument('--tau', type=float, default=0.95, metavar='G',
                    help='gae (default: 0.95)')
parser.add_argument('--l2-reg', type=float, default=1e-3, metavar='G',
                    help='l2 regularization regression (default: 1e-3)')
parser.add_argument('--ent-coef', type=float, default=0.01, 
                    help='Entropy coefficientl2 (default: 0.01)')
parser.add_argument('--learning-rate', type=float, default=3e-4, metavar='G',
                    help='learning rate (default: 3e-4)')
parser.add_argument('--clip-epsilon', type=float, default=0.2, metavar='N',
                    help='clipping epsilon for PPO')
parser.add_argument('--sac-policy',action='store_true', default=False,
                    help='use san tanh gaussian')
parser.add_argument('--optim-epochs', type=int, default=10,
                    help='epochs for the internal optimization')
parser.add_argument('--optim-batch-size', type=int, default=64,
                    help='min_batch for the internal optimzation part')
parser.add_argument('--num-threads', type=int, default=1, metavar='N',
                    help='number of threads for agent (default: 1)')
parser.add_argument('--seed', type=int, default=1, metavar='N',
                    help='random seed (default: 1)')
parser.add_argument('--min-batch-size', type=int, default=2048, metavar='N',
                    help='minimal batch size per PPO update (default: 2048)')
parser.add_argument('--max-iter-num', type=int, default=500, metavar='N',
                    help='maximal number of main iterations (default: 500)')
parser.add_argument('--log-interval', type=int, default=1, metavar='N',
                    help='interval between training status logs (default: 10)')
parser.add_argument('--save-model-interval', type=int, default=0, metavar='N',
                    help="interval between saving model (default: 0, means don't save)")
parser.add_argument('--save_path', type=str, \
                    help="path to save model pickle and log file", default='DEFAULT_DIR')
parser.add_argument('--two-losses', type=int, default=1, choices = [0,1], \
                    help="Whether to use separated losses",)
parser.add_argument('--scheduler', action='store_true', default=False, \
                    help="",)
parser.add_argument('--obs-running-state', type=int, default=1, choices = [0,1],
                    help="If the observation will be normalized (default: 1, it will be)")
parser.add_argument('--reward-running-state', type=int, default=0, choices = [0,1],
                    help="If the reward will be normalized (default: 0, it won't be)")
parser.add_argument('--reward', type=str, default='Normal',
                    help="Reward Function")
parser.add_argument('--state', type=str, default='New', choices = ['New', 'Old', 'New_Double', 'New_Double'],
                    help="State Space")
parser.add_argument('--log-prob-head', type=int, default=0, choices = [0,1],
                    help="Uses fix log prob or predicts a log_std (default: 0, it won't be)")
parser.add_argument('--gpu-index', type=int, default=0, metavar='N')


args = parser.parse_args()


if args.env_reset_mode=='False':
    args.env_reset_mode = False




dtype = torch.float64
torch.set_default_dtype(dtype)
device = torch.device('cuda', index=args.gpu_index) if torch.cuda.is_available() else torch.device('cpu')
if torch.cuda.is_available():
    torch.cuda.set_device(args.gpu_index)

print(device)
"""environment"""
env = DroneEnv(random=args.env_reset_mode,seed=args.seed, reward_function_name = args.reward, state=args.state)


action_dim = env.action_space.shape[0]
state_dim = env.observation_space[0]
# 
# 
# state_dim = env.observation_space.shape[0]
is_disc_action = len(env.action_space.shape) == 0
# running_state = ZFilter((state_dim,), clip=5)

if args.obs_running_state == 1:
    running_state = ZFilter((state_dim,), clip=3)

else:
    running_state = None

# if args.reward_running_state == 1:
#     running_reward = ZFilter((1,), demean=False, clip=10)
# else:
#     running_reward = None



# running_reward = ZFilter((1,), demean=False, clip=10)
"""seeding"""
np.random.seed(args.seed)
torch.manual_seed(args.seed)

"""define actor and critic"""
if args.model_path is None:
    if is_disc_action:
        policy_net = DiscretePolicy(state_dim, env.action_space.n)
    else:
        policy_net = Policy(state_dim, env.action_space.shape[0],\
        hidden_size=(256,256), activation='tanh', log_std=args.log_std,log_prob_head=args.log_prob_head)

    value_net = Value(state_dim, hidden_size=(256,256), activation='relu')
else:
    policy_net, value_net, running_state,  = pickle.load(open(args.model_path, "rb"))
policy_net.log_prob_head=args.log_prob_head

print(args.model_path)

policy_net.to(device)
value_net.to(device)

policy_net.log_std_min = -20
policy_net.log_std_max = 2

optimizer_policy = torch.optim.Adam(policy_net.parameters(), lr=args.learning_rate)
optimizer_value = torch.optim.Adam(value_net.parameters(), lr=args.learning_rate)

params = list(policy_net.parameters()) + list(value_net.parameters())
unique_optimizer = torch.optim.Adam(params, lr=args.learning_rate)

# optimization epoch number and batch size for PPO
optim_epochs = args.optim_epochs
optim_batch_size = args.optim_batch_size


"""create agent"""
agent = Agent(env, policy_net, device, running_state=running_state, render=args.render, num_threads=args.num_threads)





# Set save/restore paths
save_path = os.path.join('../checkpoint/', args.save_path) +'/'
check_dir(save_path)





def update_params(batch, i_iter,scheduler, scheduler_policy, scheduler_value):
    states = torch.from_numpy(np.stack(batch.state)).to(dtype).to(device)
    actions = torch.from_numpy(np.stack(batch.action)).to(dtype).to(device)
    rewards = torch.from_numpy(np.stack(batch.reward)).to(dtype).to(device)
    masks = torch.from_numpy(np.stack(batch.mask)).to(dtype).to(device)
    pre_tanh_actions = torch.from_numpy(np.stack(batch.pre_tanh_action)).to(dtype).to(device)

    with torch.no_grad():
        values = value_net(states)
        # fixed_log_probs = policy_net.get_log_prob(states, actions)[0]
        fixed_log_probs = policy_net.get_log_prob(states, pre_tanh_actions)[0]
        print("LOG PROBA mean = ", fixed_log_probs.mean())
        print("LOG PROBA std = ", fixed_log_probs.std())

    # rewards = (rewards - rewards.mean())/(rewards.std() + 1e-10)

    """get advantage estimation from the trajectories"""
    advantages, returns = estimate_advantages(rewards, masks, values, args.gamma, args.tau, device)
    # print('ADV MEAN =' ,advantages.mean())
    # print('ADV STD = ',advantages.std())
    # returns = (returns - returns.mean())/(returns.std() + 1e-10)
    # print('Returns mean = ', returns.mean())
    # print('Returns std = ', returns.std())
    # print('Values mean = ', values.mean())
    # print('Values std = ', values.std())
    with torch.no_grad():
        loss_value = (returns-values).pow(2).mean()
    print('Loss MSE mean = ',loss_value)

    print()    
    """perform mini-batch PPO update"""
    optim_iter_num = int(math.ceil(states.shape[0] / optim_batch_size))
    for epoc in range(optim_epochs):
        list_value_loss = []

        perm = np.arange(states.shape[0])
        np.random.shuffle(perm)
        perm = LongTensor(perm).to(device)
        states, actions, returns, advantages, fixed_log_probs,values = \
            states[perm].clone(), actions[perm].clone(), returns[perm].clone(), advantages[perm].clone(), fixed_log_probs[perm].clone(), values[perm].clone()
        for i in range(optim_iter_num):
            ind = slice(i * optim_batch_size, min((i + 1) * optim_batch_size, states.shape[0]))
            states_b, actions_b, advantages_b, returns_b, fixed_log_probs_b, values_b = \
                states[ind], actions[ind], advantages[ind], returns[ind], fixed_log_probs[ind], values[ind]

            
            if args.two_losses:
                policy_surr, value_loss, ev, clipfrac, entropy, approxkl = ppo_step_two_losses(policy_net, value_net, \
                optimizer_policy, optimizer_value, 1, states_b, actions_b, returns_b,
                        advantages_b, values_b, fixed_log_probs_b, args.clip_epsilon, args.l2_reg, ent_coef=args.ent_coef)

            else:
                policy_surr, value_loss, ev, clipfrac, entropy, approxkl = ppo_step_one_loss(policy_net, value_net, \
                unique_optimizer, 1, states_b, actions_b, returns_b,
                        advantages_b, values_b, fixed_log_probs_b, args.clip_epsilon, args.l2_reg, ent_coef=args.ent_coef)

            list_value_loss.append(value_loss)
        # print('Epoch = {0} | Mean = {1} | STD = {2}'.format(epoc, np.mean(list_value_loss), np.std(list_value_loss)))
    
    return policy_surr, loss_value.flatten().item(), ev, clipfrac, entropy, approxkl

def main_loop():


    list_cols = ['num_steps', 'num_episodes', 'total_reward', 'avg_reward', 'max_reward', \
        'min_reward', 'lenght_mean', 'lenght_min','lenght_max','lenght_std', 'sample_time']
        # 'min_reward', 'sample_time', 'action_mean', 'action_min', 'action_max']

    algo_cols = ['policy_loss','value_loss','explained_variance', \
                'clipfrac','entropy','aproxkl']


    writepath = os.path.join(save_path,'progress.csv')
    mode = 'a' if os.path.exists(writepath) else 'w'
    with open(writepath, mode) as outcsv:  
        writer = csv.writer(outcsv, delimiter=';', quotechar='|', quoting=csv.QUOTE_MINIMAL)   
        writer.writerow(list_cols + algo_cols) 


    nupdates = args.max_iter_num  # nupdates = total_timesteps//nbatch
    if args.scheduler:
        lr_lambda=lambda f:(1.0 - (f - 1.0) / nupdates)
    else:
        lr_lambda=lambda f: 1

    scheduler_value = torch.optim.lr_scheduler.LambdaLR(optimizer_policy, lr_lambda)
    scheduler_policy = torch.optim.lr_scheduler.LambdaLR(optimizer_value, lr_lambda)
    scheduler = torch.optim.lr_scheduler.LambdaLR(unique_optimizer, lr_lambda)

    begin=time.time()
    for i_iter in range(args.max_iter_num):
        if i_iter % 1 ==0:
            env.restart=True ## Hacky because Pyrep breaks the Drone!
            env.reset()
            env.restart=False

        print('Iter = ', i_iter)
        """generate multiple trajectories that reach the minimum batch_size"""
        batch, log = agent.collect_samples(args.min_batch_size)
        print('Done batching')
        t0 = time.time()
        loss_policy, loss_value, ev, clipfrac, entropy, approxkl = \
                    update_params(batch, i_iter, scheduler, scheduler_value, scheduler_policy)
        t1 = time.time()

        print()
        print('Loss_policy = {0:.4f}'.format(loss_policy))
        print('Loss_value = {0:.4f}'.format(loss_value))
        print('Explained variance = {0:.3f}'.format(ev))
        print('clipfrac = {0:.5f}'.format(clipfrac))
        print('entropy = {0:.5f}'.format(entropy))
        print('approxkl = {0:.5f}'.format(approxkl))
        print()

        algo_cols_values = [loss_policy, loss_value, ev, clipfrac, entropy, approxkl]


        print("Done updating")
        if i_iter % args.log_interval == 0:
            print("LOG KEYS = ", list_cols)

            print('{}\tT_sample {:.4f}\tT_update {:.4f}\tR_min {:.2f}\tR_max {:.2f}\tR_avg {:.2f}'.format(
                i_iter, log['sample_time'], t1-t0, log['min_reward'], log['max_reward'], log['avg_reward']))
            new_list=[]
            for col in list_cols:
                new_list.append(log[col])
            with open(os.path.join(save_path,'progress.csv'), 'a') as csvfile:
                rew_writer = csv.writer(csvfile, delimiter=';',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
                rew_writer.writerow(new_list + algo_cols_values)       


        if args.save_model_interval > 0 and (i_iter+1) % args.save_model_interval == 0:
            to_device(torch.device('cpu'), policy_net, value_net)
            pickle.dump((policy_net, value_net, running_state),
                        open(os.path.join(assets_dir(), '{0}/{1}_ppo_itr_{2}.p'.format(save_path,args.env_name, i_iter)), 'wb'))
            to_device(device, policy_net, value_net)


        """clean up gpu memory"""
        # torch.cuda.empty_cache()
        print('Time so far = {0:.2f} on iter = {1}'.format(time.time()-begin, i_iter))

        scheduler_value.step()
        scheduler_policy.step()
        scheduler.step()
        print(scheduler.get_last_lr())


main_loop()



