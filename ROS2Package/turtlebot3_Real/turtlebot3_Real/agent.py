from collections import deque
import numpy as np
import tensorflow as tf
import torch as T
import torch.nn as nn
from ament_index_python.packages import get_package_share_directory


class Agent:
    def __init__(self, verbose):

        package_dir = get_package_share_directory("turtlebot3_Real")
        # load weights of pretrained model


        # self.model = tf.keras.models.load_model(package_dir + "/model_trained/DDQN_id940_ep2666.h5")
        self.model = T.jit.load('/home/enange/colcon_ws/src/turtlebot3_Real/model_trained/MODEL_DQN_1000.pt') # Da sistemare
        self.model.eval()

        if verbose:
            print('==================================================')
            print('  We are using this model for the testing phase:  ')
            print('==================================================')

    def select_action(self, state):
        # we directly take the argmax for that particular state
        # action = np.argmax(self.model(state.reshape((1, -1))))

        #state = state.reshape((1, -1))
        # print("STATO:", type(state))
        #softmax_out = self.model(T.tensor(state.astype(np.float32)))
        #selected_action = np.random.choice(3, p=softmax_out.detach().numpy()[0])
        #return selected_action

        state = state.reshape(1, -1)
        s = np.array(state)
        s = s.astype(np.float32)
        action_val = self.model(T.tensor(s))
        return np.argmax(action_val.cpu().data.numpy())

        # return action

    def normalize_state(self, state):
        return np.around(state,3)