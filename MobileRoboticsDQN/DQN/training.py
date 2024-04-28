from stable_baselines3.common.logger import configure

from env.robotic_navigation import RoboticNavigation

# from alg.DDPG_PT import DDPG_PT
from alg.DQN_PT import DQN
import time, sys, argparse
import tensorflow as tf
import config
import os
from stable_baselines3 import PPO
# physical_devices = tf.config.list_physical_devices('GPU')
# for device in physical_devices:
#     tf.config.experimental.set_memory_growth(device, True)


def train(env, args):

	# Execution of the training loop
	try: 
		algo = DQN(env, verbose=2)
		algo.loop(1000)
	
	# Listener for errors and print of the eventual error message
	except Exception as e: 
		print(e)

		exc_type, exc_obj, exc_tb = sys.exc_info()
		fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
		print(exc_type, fname, exc_tb.tb_lineno)

	# In any case, close the Unity3D environment
	finally:
		env.close()

def generate_environment(editor_build, env_type):

	worker_id = int(round(time.time() % 1, 4)*10000)
	return RoboticNavigation( editor_build=editor_build, worker_id=worker_id, env_type=env_type )


# Call the main function
if __name__ == "__main__":

	# Default parameters
	args = config.parse_args()
	# seed = None implies random seed
	editor_build = True
	env_type = "training"

	print( "Mobile Robotics Lecture on ML-agents and DDPG! \n")
	env = generate_environment(editor_build, env_type)
	train(env, args)

