import gym;

gym.logger.set_level(40)
from mlagents_envs.environment import UnityEnvironment
from gym_unity.envs import UnityToGymWrapper
import numpy as np
import os, time


class RoboticNavigation(gym.Env):
    """
	A class that implements a wrapper between the Unity Engine environment of and a custom Gym environment.
	
	The main motivations for this wrapper are:

		1) Fix the sate
			originally the LiDAR scans arrive with a size of 2 * number_of_scan, beacuse for each direction Unity returns two values, the 
			first one is a float that represent the distance from the first obstacle, nomralized between [0, 1]. The second one is a flag integer [0, 1]
			which indicates if there is an obstacle in the range of the corresponing scan. To avoid a strong correlation between the sensors input of the network, 
			we removed the flag value. This is also to increase the explainability of the state value (useful also for the properties).

		2) Change the reward
			this wrapper allows us to change the reward function without modifying the Unity3D project.
	"""

    def __init__(self, step_limit=300, worker_id=0, editor_build=False, env_type="training", random_seed=0,
                 render=False):

        """
		Constructor of the class.

		Parameters
		----------
			rendered : bool
				flag to run the envornoment in rendered mode, currently unused (default: False)
		"""

        # Load the scan number given in input. Must equal 2 * RaysPerDirection + 1,
        # matching the Ray Perception Sensor Component 3D config in the Unity scene
        # (5 rays per direction -> 11 total rays, vs. the previous 3 -> 7).
        self.scan_number = 11

        # If the env_path is given as input override the environment search
        if not editor_build:

            # Sanity check for the 'env_type' option
            assert_message = "Invalid env_type, options [None, training, render]"
            assert (env_type in [None, "training", "render", "gym", "testing"]), assert_message

            # Detect the platform (linux) and load the corresponding environment path
            # without any specification using the default one.
            if env_type == "training": env_path = "env/linux_training/SafeRobotics"
            if env_type == "render": env_path = "env/linux_render/SafeRobotics"
            if env_type == "gym": env_path = "env/linux_gym/SafeRobotics"
            if env_type == "testing": env_path = "env/linux_testing/SafeRobotics"

            # If on windows override with the default parameters for it
            if os.name == "nt": "env/windows_training/SafeRobotics"

        # For the editor build force the path to None and the worker id to 0,
        # assigned values for the editor build.
        else:
            env_path = None
            worker_id = 0

        # Load the Unity Environment
        unity_env = UnityEnvironment(env_path, worker_id=worker_id, seed=random_seed)

        # Convert the Unity Environment in a OpenAI Gym Environment, setting some flag
        # according with the current setup (only one branch in output and no vision observation)
        self.env = UnityToGymWrapper(unity_env, flatten_branched=True)

        # Override the action space of the wrapper. gym_unity always reports continuous
        # actions as a symmetric Box(-1, 1) regardless of what the values actually mean
        # -- but the linear component only ever drives forward (no reverse, the LiDAR
        # can't see behind the robot; see CustomAgent.cs), so leaving it symmetric would
        # let SAC waste half its raw policy output range, and half of every random
        # warm-up action during learning_starts, on values that just collapse to
        # "stopped" in Unity. SB3 rescales the policy's canonical [-1, 1] output to
        # whatever bounds are declared here (BasePolicy.unscale_action) and samples
        # warm-up actions uniformly from this space, so declaring the real range here
        # fixes both. No change needed in step() below: UnityToGymWrapper.step()
        # forwards actions to Unity unmodified, with no validation against its own
        # (still-symmetric) action_space.
        if isinstance(self.env.action_space, gym.spaces.Box):
            self.action_space = gym.spaces.Box(
                low=np.array([0, -1], dtype=np.float32),
                high=np.array([1, 1], dtype=np.float32),
                dtype=np.float32
            )
        else:
            self.action_space = self.env.action_space

        # Override the state_size, the orginal version provide a 2*scan_number size for the LiDAR,
        # for each direction 2 value, one with the flaot value and one with the flag [0, 1]. In this
        # wrapper we remove the flag, maintaining only one value for each direction
        # Subtract also the dummy value for the cost.
        state_size = self.env.observation_space.shape[0] - self.scan_number - 1

        # Sanity check for the scan number
        assert_message = "Mismatching between the given scan number and the observations (check the cost value)"
        assert (state_size == self.scan_number + 2), assert_messages

        # Initialize the counter for the maximum step counter
        self.step_counter = 0

        # Acoording to the previous line, we override the observation space
        # lidar_scan + 2 elements, normalized in [0, 1] ==> heading (first) + distance (second)
        self.observation_space = gym.spaces.Box(
            np.array([0 for _ in range(self.scan_number)] + [0, 0]),
            np.array([1 for _ in range(self.scan_number)] + [1, 1]),
            dtype=np.float64
        )

    def reset(self):

        """
		Override of the reset function of OpenAI Gym

		Returns
		----------
			state : list
				a list of the observation, with scan_number + 2 elements, the first part contains the information
				about the ldiar sensor and the second angle and distance in respect to the target. All the values
				are normalized between [0, 1]
		"""

        # Reset the counter for the maximum step counter
        self.step_counter = 0

        # Override the state to return with the fixed state, as described in the constructor
        state = self.env.reset()

        # Call the function that fix the state according with our setup
        state = self.fix_state(state)

        # Store the distance for the reward function
        self.target_distance = state[-1]

        #
        return state

    def step(self, action):

        """
		Override of the step function of OpenAI Gym

		Parameters
		----------
			action : int
				integer that represent the action that the agent must performs

		Returns
		----------
			state : list
				a list of the observation, with scan_number + 2 elements, the first part contains the information
				about the ldiar sensor and the second angle and distance in respect to the target. All the values
				are normalized between [0, 1]
			reward : float
				a single value that represent the value of the reward function from the tuple (state, action)
			done : bool
				flag that indicates if the current state is terminal
			state : dict
				a dictionary with some additional information, currently empty
		"""

        # Call the step function of the OpenAI Gym class
        state, reward, _, _ = self.env.step(action)

        # Initialize the empty dictionary
        info = {}

        # Increase the step counter
        self.step_counter += 1

        # Computing all the info from the environment
        info["goal_reached"] = (reward == 1)
        info["collision"] = (reward == -1)
        info["cost"] = (state[-1] == 1)
        info["time_out"] = (self.step_counter >= 300)

        # Call the function that fix the state according with our setup
        state = self.fix_state(state)

        # Overrride the Done function, now from the environment we recived 'done'
        # only for the timeout
        done = (info["goal_reached"] or info["collision"] or info["time_out"])

        # Remove the reward penalty for unsupervised RL
        # if reward != 1: reward = 0

        # Here it's possible to override the reward given by the Unity Engine
        # default returns the standard reward from the environment
        reward = self.override_reward(state, reward, action, done)

        #
        return state, reward, done, info

    # Collisions are scaled up relative to Unity's raw -1: the per-step penalty below
    # (needed to stop the agent from stalling/rotating in place) also rewards shaving
    # steps off a route, so cutting a corner to save a few steps can pay off even at
    # some collision risk unless the collision penalty clearly dominates that saving.
    collision_penalty_scale = 5

    def override_reward(self, state, reward, action, done):

        # Terminal states already carry the correct value from Unity (1 for goal,
        # -1 for collision, 0 for timeout): don't shape those, only amplify collision.
        if done:
            if reward == -1:
                return reward * self.collision_penalty_scale
            return reward

        # state[-1] is now the geodesic (NavMesh) distance to the target, not the
        # straight-line one, so a corridor detour that's genuinely on the shortest
        # path decreases it like any other progress instead of being penalized.
        reward_multiplier, step_penalty = 3, 0.01
        new_distance = state[-1]
        distance_delta = self.target_distance - new_distance
        self.target_distance = new_distance

        # Clip: NavMesh corner-routing can shift discretely step to step (unlike the
        # smooth straight-line distance), so cap the shaping term to avoid single-step
        # reward spikes. step_penalty is large enough relative to this cap that idling
        # for a full episode (step_penalty * step_limit) costs more than the terminal
        # reward, removing the incentive to rotate in place instead of committing.
        return np.clip(distance_delta * reward_multiplier, -0.05, 0.05) - step_penalty

    def fix_state(self, state):

        """
		Support function to convert the observation vector from the version obtained by Unity3D to our configuration.
		The orginal version provide a 2*scan_number size for the LiDAR,
		for each direction 2 value, one with the flaot value and one with the flag [0, 1]. 
		In this	wrapper we remove the flag, maintaining only one value for each direction

		Parameters
		----------
			state : list
				a list of the observation original observations from the environment

		Returns
		----------
			state : list
				a list of the observation, with scan_number + 2 elements, the first part contains the information
				about the ldiar sensor and the second angle and distance in respect to the target. All the values
				are normalized between [0, 1]
		"""

        # Compute the size of the observation array that correspond to the lidar sensor,
        # the other portion is maintened
        scan_limit = 2 * (self.scan_number)
        state_lidar = [s for id, s in enumerate(state[:scan_limit]) if id % 2 == 1]

        # Change the order of the lidar scan to the order of the wrapper (see the class declaration for details)
        lidar_ordered_1 = [s for id, s in enumerate(reversed(state_lidar)) if id % 2 == 0]
        lidar_ordered_2 = [s for id, s in enumerate(state_lidar) if id % 2 == 1]
        lidar_ordered = lidar_ordered_1 + lidar_ordered_2

        # Concatenate the ordered lidar state with the other values of the state
        state_fixed = lidar_ordered + list(state[scan_limit:-1])

        #
        return np.array(state_fixed)

    # Override the "close" function
    def close(self):
        self.env.close()

    # Override the "render" function
    def render(self):
        pass
