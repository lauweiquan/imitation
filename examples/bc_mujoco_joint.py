"""
Used to train and load BC policy for simulated data in MujoCo with joint pose as action
"""

import numpy as np
from stable_baselines3.common.evaluation import evaluate_policy
from imitation.data.types import Trajectory, DictObs

from imitation.algorithms import bc
from imitation.data import rollout
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
    image_transpose = np.transpose(image_array, (1, 0))
    return image_transpose

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
    image_transpose = np.transpose(image_array, (2, 1, 0))
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
                    'cart_traj': [],
                    'joint_pose': []
                    }
    traj_list = []
    actions = []
    infos = []
    terminals = []
    delta_list = []
    cart_list = []
    joint_list = []
    ft_list = []
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
        for cart_traj in data[2].strip(' ').strip('[').strip(']').split(' '):
            cart_list.append(float(cart_traj))
        for joint_traj in data[3].strip(' ').strip('[').strip(']').split(' '):
            joint_list.append(float(joint_traj))
        observations['depth_image'].append(depth_image)
        observations['cart_traj'].append(pd.eval(cart_list))
        actions.append(np.array(pd.eval(joint_list), dtype="float32"))
        terminals.append(data[4].strip())
        cart_list = []
        joint_list = []

# ----------------------------------------------------------------------------------
        # toggle ft and color by commenting (need to commend observation dict part)
        color_path = data[1].strip()
        color_image = load_color_image(color_path)
        observations['color_image'].append(color_image)
        for ft in data[5].strip('\n').strip(' ').strip('[').strip(']').split(' '):
            ft_list.append(float(ft))
        observations['ft_sensor'].append(pd.eval(ft_list))
        ft_list = []
# ----------------------------------------------------------------------------------

        # convert to delta cart traj
        if i == 0:
            pass
        else:
            delta = actions[i] - actions[i-1]
            delta_list.append(delta)
            infos.append(None)
        i += 1       
 
    return traj_list
 
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
    cart_list = []
    joint_list = []
    ft_list = []
    for line in lines:
        if line[0] == "\\":
            break

        data = line.split('|')
        depth_path = data[0].strip()
        depth_image = load_depth_image(depth_path)
        observations['depth_image'] = depth_image
        for cart_traj in data[2].strip(' ').strip('[').strip(']').split(' '):
            cart_list.append(float(cart_traj))
        for joint_traj in data[3].strip(' ').strip('[').strip(']').split(' '):
            joint_list.append(float(joint_traj))
        observations['cart_traj'] = pd.eval(cart_list)
        observations['joint_pose'] = pd.eval(joint_list)
        actions.append(np.array(pd.eval(joint_list), dtype="float32"))
        cart_list = []
        joint_list = []
    # ------------------------------------------------------------------------------------------
        # toggle ft and color data by commenting (need to comment obs dict part)
        color_path = data[1].strip()
        color_image = load_color_image(color_path)
        observations['color_image'] = color_image
        for ft in data[5].strip('\n)').strip(' ').strip('[').strip(']').split(' '):
            ft_list.append(float(ft))
        observations['ft_sensor'] = pd.eval(ft_list)
        ft_list = []
    # ------------------------------------------------------------------------------------------
        acts, _ = policy.predict(observations, deterministic = True)
        # print("prediction")
        print(np.array2string(acts, separator = ","), ",")
        acts_list.append(acts)
    # for act in acts_list:
    #     print(act, ",")
    return acts_list

if __name__ == "__main__":
    # Update these values to match your image size
    H = 320
    W = 240


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
    trajectory_file_path = '../data/space_data_130.txt'
    trajectory = load_trajectory_from_file(trajectory_file_path)
    transitions = rollout.flatten_trajectories(trajectory)

    # train BC policy
    bc_trainer = bc.BC(
        observation_space= observation_space,
        action_space= action_space,
        demonstrations=transitions,
        rng=rng,
    )
    bc_trainer.train(n_epochs=3)
    save_policy(bc_trainer.policy,"../trained_policy/feeding_policy_mujoco_save")

    # Load trained policy and sample observation data
    loaded_policy = bc.reconstruct_policy("../trained_policy/feeding_policy_mujoco_test")
    obs_file_path = '../data_sim/sample_data.txt'

    # predict actions based on observation for trained policy
    acts = load_predict_observation(obs_file_path, loaded_policy)


    # reward not used for BC
    # reward, _ = evaluate_policy(bc_trainer.policy, env, 10)
    # print("Reward:", reward)
