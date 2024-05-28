"""
Used to train and load BC policy for real world data
"""

import numpy as np
import gymnasium as gym
from stable_baselines3.common.evaluation import evaluate_policy
from imitation.data.types import Trajectory, DictObs

from imitation.algorithms import bc
from imitation.data import rollout
from imitation.util.util import make_vec_env, save_policy

from gymnasium import spaces
from PIL import Image
import pandas as pd

# custom gym env
from typing import Dict, Optional
from typing import Any
import numpy as np

from gymnasium.wrappers import TimeLimit
from imitation.data.wrappers import RolloutInfoWrapper
from imitation.util.util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv


class ObservationMatchingEnv(gym.Env):
    def __init__(self):
        self.state = None
        # Update these values to match your image size
        H = 480
        W = 640

        # setup observation space dict
        self.observation_space = spaces.Dict(
                        {
                            # Modify accordingly if you need to use joint states
                            "cart_traj": spaces.Box(low=-1.0, 
                                                    high=1.0, 
                                                    shape=(7,), 
                                                    dtype=np.float64),

                            "joint_pose": spaces.Box(low=-1.0, 
                                            high=1.0, 
                                            shape=(6,), 
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

        self.action_space = spaces.Box(low=-1.0, 
                                    high=1.0, 
                                    shape=(6,), 
                            dtype=np.float64)

    def reset(self, seed: int = None, options: Optional[Dict[str, Any]] = None):
        super().reset(seed=seed, options=options)
        self.state = self.observation_space.sample()
        return self.state, {}

    def step(self, action):
        reward = None
        self.state = self.observation_space.sample()
        return self.state, reward, False, False, {}

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
    return image_transpose

# Load trajectory data from a text file to train BC policy
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
                    'cart_traj': [],
                    'joint_pose': []
                    }
    traj_list = []
    actions = []
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
            observations['joint_pose'] = np.array(actions)
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
                    'cart_traj': [],
                    'joint_pose': []
                    }
            actions = []
            delta_list = []
            infos = []
            terminals = []
            i = 0
            continue

        data = line.split('|')
        depth_path = data[0].strip()
        depth_image = load_depth_image(depth_path)
        color_path = data[1].strip()
        color_image = load_color_image(color_path)
        observations['depth_image'].append(depth_image)
        observations['color_image'].append(color_image)
        observations['ft_sensor'].append(pd.eval(data[5]))
        observations['cart_traj'].append(pd.eval(data[2]))
        actions.append(np.array(pd.eval(data[3]), dtype="float32"))
        terminals.append(data[4].strip())

        # convert to delta cart traj
        # initial point not moving hence pass
        if i == 0:
            pass
        else:
            delta = actions[i] - actions[i-1]
            delta_list.append(delta)
            infos.append(None)
            
        i += 1       
 
    return traj_list

# predict action based on sample observation
def load_predict_observation(file_path, policy):
    # Load trajectory data from the text file
    with open(file_path, 'r') as file:
        lines = file.readlines()
 
    # Parse trajectory data
    observations = {
                    'depth_image': [],
                    'color_image': [],
                    'ft_sensor': [],
                    'cart_traj': [],
                    'joint_pose': []
                    }
    acts_list = []
    actions = []
    for line in lines:
        if line[0] == "\\":
            break

        data = line.split('|')
        depth_path = data[0].strip()
        depth_image = load_depth_image(depth_path)
        color_path = data[1].strip()
        color_image = load_color_image(color_path)
        observations['depth_image'] = depth_image
        observations['color_image'] = color_image
        observations['ft_sensor'] = pd.eval(data[5])
        observations['cart_traj'] = pd.eval(data[2])
        observations['joint_pose'] = pd.eval(data[3])
        actions.append(np.array(pd.eval(data[3]), dtype="float32"))
        
        acts, _ = policy.predict(observations, deterministic = True)
        print(np.array2string(acts, separator = ","), ",")
        acts_list.append(acts)
    return acts_list
    

# Update these values to match your image size
H = 480
W = 640


# Define the observation and action spaces
observation_space = spaces.Dict(
                {
                    # Modify accordingly if you need to use joint states
                    "cart_traj": spaces.Box(low=-1.0, 
                                            high=1.0, 
                                            shape=(7,), 
                                            dtype=np.float64),
                                            
                    "joint_pose": spaces.Box(low=-1.0, 
                                            high=1.0, 
                                            shape=(6,), 
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

rng = np.random.default_rng(0)

# read trajectory data from file and convert to transitions format
trajectory_file_path = '../data/space_data.txt'
trajectory = load_trajectory_from_file(trajectory_file_path)
transitions = rollout.flatten_trajectories(trajectory)

# train BC policy and save it
bc_trainer = bc.BC(
    observation_space= observation_space,
    action_space= action_space,
    demonstrations=transitions,
    rng=rng,
)
bc_trainer.train(n_epochs=22)
save_policy(bc_trainer.policy,"../trained_policy/feeding_policy")

# Load trained policy and sample observation data
loaded_policy = bc.reconstruct_policy("../trained_policy/feeding_policy_real")
obs_file_path = '../data/sample_data.txt'

# predict actions based on observation for trained policy
acts = load_predict_observation(obs_file_path, loaded_policy)

# reward not used for BC
# reward, _ = evaluate_policy(loaded_policy, env, 10)
# print("Reward:", reward)

