import numpy as np
from stable_baselines3.common.evaluation import evaluate_policy
from imitation.data.types import Trajectory, DictObs

from imitation.algorithms import bc
from imitation.data import rollout
from imitation.data.wrappers import RolloutInfoWrapper
from imitation.policies.serialize import load_policy
from imitation.util.util import make_vec_env
from imitation.data.types import DictObs

from gymnasium import spaces
from PIL import Image
import pandas as pd

def load_image(image_path):
    """
    Load an image from file and convert it to a numpy array.
 
    Args:
        image_path (str): Path to the image file.
 
    Returns:
        np.ndarray: Numpy array containing the image data.
    """
    image = Image.open(image_path)
    image_array = np.array(image)
    # print("printing image array")
    # print(image_array)
    return image_array

 
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
                    'cart_traj': []
                    }
    trajectory_data = {
        'obs': [],
        'acts': [],
        'infos': [],
        'terminal': []
    }
    one_traj_list = []
    all_traj_list = []
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
            # observations['depth_image'] = np.array(observations['depth_image'])
            # observations['color_image'] = np.array(observations['color_image'])
            # observations['cart_traj'] = np.array(observations['cart_traj'])
            # print(observations['color_image'].shape)
            # trajectory_data = {
            #     'obs': DictObs(observations),
            #     'acts': np.array(actions),
            #     'infos': infos,
            #     'terminal': terminals
            # }
            # traj_list.append(Trajectory(**trajectory_data))
            all_traj_list.append(one_traj_list)
            observations = {
                    'depth_image': [],
                    'color_image': [],
                    'cart_traj': []
                    }
            trajectory_data = {
                'obs': [],
                'acts': [],
                'infos': [],
                'terminal': []
            }
            actions = []
            infos = []
            terminals = []
            one_traj_list = []
            i = 0
            continue

        data = line.split('|')
        # print(data)
        depth_path = data[0].strip()
        depth_image = load_image(depth_path)
        color_path = data[1].strip()
        color_image = load_image(color_path)
        # print(depth_image)
        # observations['depth_image'].append(depth_image)
        # observations['color_image'].append(color_image)
        # observations['cart_traj'].append(pd.eval(data[2]))
        # print(observations)
        # obs_list.append(DictObs(observations))
        actions.append(np.array(pd.eval(data[3])))
        terminals.append(data[4].strip())
        # rewards.append(float(data[2]))
        # dones.append(bool(int(data[3])))
        observations['depth_image'] = np.array(depth_image)
        observations['color_image'] = np.array(color_image)
        # print(observations['color_image'].shape)
        observations['cart_traj'] = np.array(eval(data[2]))
        # for test in observations.items():
        #     print(test)
        trajectory_data['obs'] = DictObs(
                                {
                                 "depth_image": observations['depth_image'],
                                 "color image": observations['color_image'],
                                 "cart traj": observations['cart_traj']
                                }
                            )
        

        # convert to delta cart traj
        if i == 0:
            # trajectory_data['acts'] = []
            # trajectory_data['infos'] = None
            pass
        else:
            # print(type(act))
            delta = actions[i] - actions[i-1]
            # print(delta)
            # delta_list.append(delta)
            # actions.append(pd.eval(data[3]))
            # infos.append(None)
            trajectory_data['acts'] = delta
            trajectory_data['infos'] = None
        i += 1       
 
        # trajectory_data['obs'] = DictObs(observations)
        # trajectory_data['acts'] = np.array(pd.eval(data[3].strip()))
        # trajectory_data['infos'] = None
        trajectory_data['terminal'] = [(data[4].strip())]
        one_traj_list.append(Trajectory(**trajectory_data))

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
 
    return all_traj_list
 

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

                    "depth_image": spaces.Box(low=0,
                                        high=255,
                                        shape=(H, W),
                                        dtype=np.uint8),

                    "color_image": spaces.Box(low=0,
                                        high=255,
                                        shape=(H, W, 3),
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
transitions = load_trajectory_from_file(trajectory_file_path)
# print(trajectory)
# transitions = rollout.flatten_trajectories(trajectory)
# transitions = rollout.flatten_trajectories(rollouts)
# for roll in rollouts:
#     print(len(roll))
# for elements in transitions:
#     print(elements)
data = np.load('transitions.npy', allow_pickle=True)
# print(data)
bc_trainer = bc.BC(
    observation_space= observation_space,
    action_space= action_space,
    demonstrations=transitions,
    rng=rng,
)
bc_trainer.train(n_epochs=1)
# reward, _ = evaluate_policy(bc_trainer.policy, env, 10)
# print("Reward:", reward)