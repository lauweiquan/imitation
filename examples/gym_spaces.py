from gymnasium import spaces
import numpy as np

# Update these values to match your image size
H = 480
W = 640


# This example below assumes your image is a RGB image (no depth)
# If you want to include depth, you can change shape to be (H, W, 4) and modify your data accordingly
observation_space = spaces.Dict(
                {
                    # Modify accordingly if you need to use joint states
                    "eef_pose": spaces.Box(low=-1.0, 
                                            high=1.0, 
                                            shape=(3,), 
                                            dtype=np.float64),

                    "image": spaces.Box(low=0,
                                        high=255,
                                        shape=(H, W, 3),
                                        dtype=np.uint8),
                }
            )


action_space = spaces.Dict(
                {
                    "joint_positions": spaces.Box(low=-1.0, 
                                                  high=1.0, 
                                                  shape=(6,), 
                                                  dtype=np.float64),
                }
            )