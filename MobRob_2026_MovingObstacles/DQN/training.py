from env.robotic_navigation import RoboticNavigation
from alg.DDQN import DDQN
from alg.PPO_SB3 import PPO_SB3
from alg.SAC_SB3 import SAC_SB3
import time, sys, argparse
import config

ALGORITHMS = {
	"DDQN": DDQN,
	"PPO_SB3": PPO_SB3,
	"SAC_SB3": SAC_SB3,
}

def train(env, args):

	# Execution of the training loop
	try:
		algo = ALGORITHMS[args.alg](env, args)
		algo.loop(args)
	# Listener for errors and print of the eventual error message
	except Exception as e:
		print(e)

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

	print( "Mobile Robotics Lecture on ML-agents and DDQN! \n")
	env = generate_environment(editor_build, env_type)
	train(env, args)


