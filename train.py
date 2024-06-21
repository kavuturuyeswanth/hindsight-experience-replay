import numpy as np
import gymnasium as gym
from gymnasium.envs.registration import register
import os, sys
from arguments import get_args
from mpi4py import MPI
from rl_modules.ddpg_agent import ddpg_agent
import random
import torch

"""try:
    import panda_gym
except ImportError:
    pass
"""
try:
    import gymnasium_robotics
except ImportError:
    pass

import sys
sys.path.append('/misis/project_multi_skill_rl/rl_skills/envs')
import mujoco_UR3e_reach
import ur3e_pick_and_place
import mujoco_UR3e_slide
import mujoco_UR3e_push

for reward_type in ["sparse", "dense"]:
    suffix = "Dense" if reward_type == "dense" else ""
    register(
        id=f"UR3eReach{suffix}-v1",
        entry_point="mujoco_UR3e_reach:UR3eReachEnv",
        kwargs={"reward_type": reward_type,},
        max_episode_steps=50,
        )

    register(
        id=f"UR3ePickAndPlace{suffix}-v1",
        entry_point="ur3e_pick_and_place:MujocoUR3ePickAndPlaceEnv",
        kwargs={"reward_type": reward_type, "render_mode": "rgb_array"},
        max_episode_steps=50,
        )
    register(
        id=f"UR3ePush{suffix}-v1",
        entry_point="mujoco_UR3e_push:UR3ePushEnv",
        kwargs={"reward_type": reward_type,},
        max_episode_steps=50,
        )        
    register(
        id=f"UR3eSlide{suffix}-v1",
        entry_point="mujoco_UR3e_slide:UR3eSlideEnv",
        kwargs={"reward_type": reward_type,},
        max_episode_steps=50,
        )

"""
train the agent, the MPI part code is copy from openai baselines(https://github.com/openai/baselines/blob/master/baselines/her)

"""
def get_env_params(env,seed):
    obs,_ = env.reset(seed=seed)
    # close the environment
    params = {'obs': obs['observation'].shape[0],
            'goal': obs['desired_goal'].shape[0],
            'action': env.action_space.shape[0],
            'action_max': env.action_space.high[0],
            }
    params['max_timesteps'] = env._max_episode_steps
    
    return params

def launch(args):
    # create the ddpg_agent
    env = gym.make(args.env_name)
    # set random seeds for reproduce
    
    random.seed(args.seed + MPI.COMM_WORLD.Get_rank())
    np.random.seed(args.seed + MPI.COMM_WORLD.Get_rank())
    torch.manual_seed(args.seed + MPI.COMM_WORLD.Get_rank())
    if args.cuda:
        torch.cuda.manual_seed(args.seed + MPI.COMM_WORLD.Get_rank())
    # get the environment parameters
    env_seed = args.seed + MPI.COMM_WORLD.Get_rank()
    env_params = get_env_params(env, env_seed)
    # create the ddpg agent to interact with the environment 
    ddpg_trainer = ddpg_agent(args, env, env_params)
    ddpg_trainer.learn()

if __name__ == '__main__':
    # take the configuration for the HER
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['IN_MPI'] = '1'
    # get the params
    args = get_args()
    launch(args)
