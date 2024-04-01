import numpy as np
# import gymnasium as gym
from stable_baselines3.common.evaluation import evaluate_policy
from imitation.data.types import Trajectory, DictObs

from imitation.algorithms import bc
from imitation.data import rollout
from imitation.data.wrappers import RolloutInfoWrapper
from imitation.policies.serialize import load_policy
from imitation.util.util import make_vec_env, save_policy

from gymnasium import spaces
from PIL import Image
import pandas as pd

def load_depth_image(image_path):
    """
    Load an image from file and convert it to a numpy array.
 
    Args:
        image_path (str): Path to the image file.
 
    Returns:
        np.ndarray: Numpy array containing the image data.
    """
    image = Image.open(image_path)
    image_array = np.array(image)
    # print("printing depth image array")
    # print(image_array.shape)
    return image_array

def load_color_image(image_path):
    """
    Load an image from file and convert it to a numpy array.
 
    Args:
        image_path (str): Path to the image file.
 
    Returns:
        np.ndarray: Numpy array containing the image data.
    """
    image = Image.open(image_path)
    image_array = np.array(image)
    image_transpose = np.transpose(image_array, (2, 0, 1))
    # print("printing color image array")
    # print(image_transpose.shape)
    return image_transpose

 
def load_trajectory_from_file(file_path):
    """
    Load trajectory data from a text file.
 
    Args:
        file_path (str): Path to the text file containing the trajectory data.
 
    Returns:
        Trajectory: Trajectory object containing the loaded data.
    """
    # Load trajectory data from the text file
    with open(file_path, 'r') as file:
        lines = file.readlines()
 
    # Parse trajectory data
    observations = {
                    'depth_image': [],
                    'color_image': [],
                    'ft_sensor': [],
                    'cart_traj': []
                    }
    # trajectory_data = {
    #     'obs': [],
    #     'acts': [],
    #     'infos': [],
    #     'terminal': []
    # }
    traj_list = []
    obs_list = []
    actions = []
    rewards = []
    dones = []
    infos = []
    terminals = []
    delta_list = []
    i = 0
    for line in lines:
        if line[0] == "\\":
            observations['depth_image'] = np.array(observations['depth_image'])
            observations['color_image'] = np.array(observations['color_image'])
            observations['ft_sensor'] = np.array(observations['ft_sensor'])
            observations['cart_traj'] = np.array(observations['cart_traj'], dtype="float32")
            # print(observations['color_image'].shape)
            trajectory_data = {
                'obs': DictObs(observations),
                'acts': np.array(delta_list),
                'infos': None,
                'terminal': True
            }
            traj_list.append(Trajectory(**trajectory_data))
            observations = {
                    'depth_image': [],
                    'color_image': [],
                    'ft_sensor': [],
                    'cart_traj': []
                    }
            actions = []
            delta_list = []
            infos = []
            terminals = []
            i = 0
            continue

        data = line.split('|')
        # print(data)
        depth_path = data[0].strip()
        depth_image = load_depth_image(depth_path)
        color_path = data[1].strip()
        color_image = load_color_image(color_path)
        # print(depth_image)
        observations['depth_image'].append(depth_image)
        observations['color_image'].append(color_image)
        observations['ft_sensor'].append(pd.eval(data[5]))
        observations['cart_traj'].append(pd.eval(data[2]))
        # print(observations)
        # obs_list.append(DictObs(observations))
        actions.append(np.array(pd.eval(data[3]), dtype="float32"))
        terminals.append(data[4].strip())
        # rewards.append(float(data[2]))
        # dones.append(bool(int(data[3])))

        # convert to delta cart traj
        if i == 0:
            # delta_list.append(pd.eval(data[3]))
            # trajectory_data['acts'] = []
            # trajectory_data['infos'] = None
            pass
        else:
            # print(type(act))
            delta = actions[i] - actions[i-1]
            # print(delta)
            # delta_list.append(delta)
            delta_list.append(delta)
            infos.append(None)
            # trajectory_data['acts'] = delta
            # trajectory_data['infos'] = None
        i += 1       
 
        # trajectory_data['obs'] = DictObs(observations)
        # trajectory_data['acts'] = np.array(pd.eval(data[3].strip()))
        # trajectory_data['infos'] = None
        # trajectory_data['terminal'] = [(data[4].strip())]

        # print("trajectory no: ", i)
        # print(trajectory_data)

        # traj_list.append(Trajectory(**trajectory_data))        
    # print(traj_list)
        
    # convert to delta cart traj
    # i = 0
    # for act in actions:
    #     if i == 0:
    #         pass
    #     else:
    #         # print(type(act))
    #         delta = actions[i] - actions[i-1]
    #         # print(delta)
    #         delta_list.append(delta)
    #         infos.append(None)
    #     i += 1

    # print(observations)
        
    # observations['depth_image'] = np.array(observations['depth_image'])
    # observations['color_image'] = np.array(observations['color_image'])
    # observations['cart_traj'] = np.array(observations['cart_traj'])
    
    # trajectory_data = {
    #     'obs': DictObs(observations),
    #     'acts': np.array(actions),
    #     'infos': infos,
    #     'terminal': terminals
    # }
    # print("traj data", trajectory_data)
    # print("converted traj",Trajectory(**trajectory_data))
 
    return traj_list
 

# Update these values to match your image size
H = 480
W = 640


# This example below assumes your image is a RGB image (no depth)
# If you want to include depth, you can change shape to be (H, W, 4) and modify your data accordingly
observation_space = spaces.Dict(
                {
                    # Modify accordingly if you need to use joint states
                    "cart_traj": spaces.Box(low=-1.0, 
                                            high=1.0, 
                                            shape=(7,), 
                                            dtype=np.float64),

                    "ft_sensor": spaces.Box(low=-1.0, 
                                            high=1.0, 
                                            shape=(6,), 
                                            dtype=np.float64),

                    "depth_image": spaces.Box(low=0,
                                        high=255,
                                        shape=(H, W),
                                        dtype=np.uint8),

                    "color_image": spaces.Box(low=0,
                                        high=255,
                                        shape=(3, H, W),
                                        dtype=np.uint8)
                }
            )

action_space = spaces.Box(low=-1.0, 
                            high=1.0, 
                            shape=(6,), 
                            dtype=np.float64)

# action_space = spaces.Dict(
#                 {
#                     "joint_positions": spaces.Box(low=-1.0, 
#                                                   high=1.0, 
#                                                   shape=(6,), 
#                                                   dtype=np.float64)
#                 }
#             )

# print(observation_space)
# print(action_space)

rng = np.random.default_rng(0)

# venv = make_vec_env("your-env", n_envs=4, rng=np.random.default_rng())
# env = make_vec_env(
#     "seals:seals/CartPole-v0",
#     rng=rng,
#     n_envs=1,
#     post_wrappers=[lambda env, _: RolloutInfoWrapper(env)],  # for computing rollouts
# )
# expert = load_policy(
#     "ppo-huggingface",
#     organization="HumanCompatibleAI",
#     env_name="seals-CartPole-v0",
#     venv=env,
# )
# rollouts = rollout.rollout(
#     expert,
#     env,
#     rollout.make_sample_until(min_timesteps=None, min_episodes=50),
#     rng=rng,
# )
# Example usage:
trajectory_file_path = '../data/space_data.txt'
trajectory = load_trajectory_from_file(trajectory_file_path)
# for traj in trajectory:
#     print(traj)
transitions = rollout.flatten_trajectories(trajectory)
# transitions = rollout.flatten_trajectories(rollouts)
# for roll in rollouts:
#     print(len(roll))
# print(rollouts)
# for elements in transitions:
#     print(elements)
#     print("one element done")
# data = np.load('transitions.npy', allow_pickle=True)
# for dee in data:
#     print(dee['obs'].shape)
#     print("one element done")
bc_trainer = bc.BC(
    observation_space= observation_space,
    action_space= action_space,
    demonstrations=transitions,
    rng=rng,
)
bc_trainer.train(n_epochs=21)
save_policy(bc_trainer.policy,"feeding_policy")
loaded_policy = bc.reconstruct_policy("feeding_policy")
# reward, _ = evaluate_policy(bc_trainer.policy, env, 10)
# print("Reward:", reward)
