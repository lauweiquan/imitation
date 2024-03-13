import numpy as np
import gymnasium as gym
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor, VecNormalize, VecTransposeImage, VecVideoRecorder

from environments.reaching_rgb_env import XArmReachRGBEnv

from imitation.algorithms import bc
from imitation.data import rollout
from imitation.data.wrappers import RolloutInfoWrapper
from imitation.policies.serialize import load_policy
from imitation.util.util import make_vec_env

from controllers import load_controller_config


model_path = "models/robots/xarm6/xarm6_with_ft_sensor_gripper_with_spoon.xml"

config = {# Simulation
            'model_path': model_path,
            'sim_timestep': 0.01,
            'camera': "d435",       # can be a camera ID or string (camera name)
            'obs_mode': "rgb_array",

            }

controller_config = load_controller_config(default_controller="OSC_POSE")


rng = np.random.default_rng(0)

kwargs = config
kwargs.update({"controller_config": controller_config})
kwargs.update({"model_path": model_path})

env = make_vec_env(
    "XArmReachRGBEnv-v0",
    rng=rng,
    n_envs=1,
    post_wrappers=[lambda env, _: RolloutInfoWrapper(env)],  # for computing rollouts
    env_make_kwargs=kwargs,
)
print(env.observation_space)
env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0, clip_reward=10.0)
env = VecTransposeImage(env)
print(env.observation_space)

print(">> Loading policy")
expert = load_policy(
    policy_type="ppo",
    path="logs/XArmReachRGBEnv-v0/ppo_latest_model/model",
    venv=env,
)

print(">> Generating rollouts")
rollouts = rollout.rollout(
    expert,
    env,
    rollout.make_sample_until(min_timesteps=None, min_episodes=5),
    rng=rng,
)

print(">> Rolling out")

transitions = rollout.flatten_trajectories(rollouts)

print("Transitions:", type(transitions))

# save transitions into a txt file
np.save("/home/janne/Desktop/transitions", transitions, allow_pickle=True)

# To load this np file
# transitions = np.load("/home/janne/Desktop/transitions.npy", allow_pickle=True)