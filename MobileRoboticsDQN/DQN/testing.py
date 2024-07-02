import warnings; warnings.filterwarnings("ignore")
import os; os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import sys; sys.path.append("./")
from env.robotic_navigation import RoboticNavigation
import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU') 
for device in physical_devices:
    tf.config.experimental.set_memory_growth(device, True)
import numpy as np
import torch as T
from stable_baselines3 import PPO
import time



FLAG = True

def get_action( state, policy ):

	# if state[-2] < 0.01: state[-2] = 1

	action = state.reshape((1,-1))
	softmax_out = policy(T.tensor(action)).cpu().data.numpy()
	selected_action = np.argmax( softmax_out )
	print(selected_action)
	return selected_action


def main( env, policy_network, iterations=100 ):

	goal, crash = 0, 0

	for ep in range(iterations):

		state = env.reset()

		while True:
			action = get_action( state, policy_network )
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

	# policy_network = T.jit.load("/home/riccardo/Desktop/TurtleBot/MobileRoboticsDQN/DQN/model_testing/MODEL_DQN_500.pt")
	# policy_network.eval()
	policy_network = PPO.load("/home/riccardo/Desktop/TurtleBot/MobileRoboticsDQN/DQN/model_testing/ppo_5000_baseline3")


	try:
		env = RoboticNavigation(env_type= "testing", editor_build=False,random_seed=123 )
		success = main( env, policy_network )
		print('\n======================================')
		print(f'\nSuccess: {success[0]}/{success[2]}\nCrash: {success[1]}/{success[2]}\n')
		print('======================================\n')

	finally:
		env.close()
