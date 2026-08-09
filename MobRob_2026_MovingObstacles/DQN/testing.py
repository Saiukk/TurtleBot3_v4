import warnings; warnings.filterwarnings("ignore")
import os; os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import sys; sys.path.append("./")
import argparse
import gym
from env.robotic_navigation import RoboticNavigation
import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU')
for device in physical_devices:
    tf.config.experimental.set_memory_growth(device, True)
from stable_baselines3 import PPO, SAC
import numpy as np
import time


FLAG = True

def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument("--alg", type=str, default="DDQN", choices=["DDQN", "PPO_SB3", "SAC_SB3"], help="Algorithm the checkpoint was trained with")
	parser.add_argument("--model_path", type=str, required=True, help="Path to the trained model checkpoint (.h5 for DDQN, .zip for PPO_SB3/SAC_SB3)")
	parser.add_argument("--iterations", type=int, default=100, help="Number of test episodes")
	parser.add_argument("--epsilon", type=float, default=0.05, help="Probability of taking a uniformly random action instead of the policy's choice, to break deterministic action cycles (0 disables)")
	return parser.parse_args()


def fix_heading_edge_case(state):
	# Tuned around the heading encoding's wraparound at 0/1 for "target directly
	# behind" (see robotic_navigation.py / CustomAgent.cs): remaps one edge case to
	# the other representation. Kept for both algorithms for parity, but worth
	# re-validating empirically against freshly trained checkpoints.
	if state[-2] < 0.01: state[-2] = 1
	return state


def get_action_ddqn( state, policy, env ):
	state = fix_heading_edge_case(state)
	q_values = policy(state.reshape((1, -1))).numpy()
	return int(np.argmax(q_values))


def get_action_sb3( state, policy, env ):
	state = fix_heading_edge_case(state)
	# deterministic=False samples from the policy's own learned action distribution
	# instead of always taking the argmax -- matches how actions were actually
	# selected during training rollouts, and gives a first line of defense against
	# the deterministic policy repeating the same action forever in a tied/cyclic
	# state. See --epsilon for a guaranteed floor of randomness on top of this.
	action, _ = policy.predict(state, deterministic=False)
	# PPO_SB3 can be Discrete(3) or Continuous(2) depending on the Unity scene's
	# useContinuousActions toggle; SAC_SB3 is always continuous. Discrete needs a
	# plain int for env.step(); continuous keeps its array shape.
	if isinstance(env.action_space, gym.spaces.Discrete):
		return int(np.asarray(action).item())
	return np.asarray(action)


def main( env, policy_network, get_action, iterations=100, epsilon=0.0 ):

	goal, crash = 0, 0

	for ep in range(iterations):

		state = env.reset()

		while True:
			if np.random.random() < epsilon:
				# Works for both Discrete and Box action spaces
				action = env.action_space.sample()
			else:
				action = get_action( state, policy_network, env )
			state, _, done, info = env.step(action)
			if done: break


		if info["goal_reached"]:
			print( f"{ep:3}: Goal!" )
			goal += 1

		elif info["collision"]:
			print( f"{np.round(state, 4)} => {action}")
			print( f"{ep:3}: Crash!" )
			crash += 1

		else:
			print( f"{ep:3}: Time Out!" )

	return goal, crash, iterations


if __name__ == "__main__":

	args = parse_args()

	if args.alg == "DDQN":
		policy_network = tf.keras.models.load_model(args.model_path)
		get_action = get_action_ddqn
	elif args.alg == "PPO_SB3":
		policy_network = PPO.load(args.model_path)
		get_action = get_action_sb3
	else:
		policy_network = SAC.load(args.model_path)
		get_action = get_action_sb3

	try:
		env = RoboticNavigation(env_type= "testing", editor_build=True )
		success = main( env, policy_network, get_action, iterations=args.iterations, epsilon=args.epsilon )
		print('\n======================================')
		print(f'\nSuccess: {success[0]}/{success[2]}\nCrash: {success[1]}/{success[2]}\n')
		print('======================================\n')

	finally:
		env.close()
