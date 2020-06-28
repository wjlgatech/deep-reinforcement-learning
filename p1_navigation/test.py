# to activate UnityML environment in terminal, type:
# source activate /Users/jialiangwu/anaconda3/envs/UnityML

from unityagents import UnityEnvironment
import numpy as np
from dqnAgent import Agent
import torch

env = UnityEnvironment(file_name="Banana.app")

# get the default brain
brain_name = env.brain_names[0]
brain = env.brains[brain_name]


agent = Agent(state_size=brain.vector_observation_space_size, action_size=brain.vector_action_space_size, seed=0)

# Load trained model weights
agent.qnetwork_local.load_state_dict(torch.load('dqnAgent_Trained_Model.pth'))

env_info = env.reset(train_mode=False)[brain_name]  # reset the environment
state = env_info.vector_observations[0]            # get the current state
score = 0
eps = 0.0

# initialize the score
while True:
    #replace: action = np.random.randint(action_size)        # select an action
    action = agent.act(state, eps)
    env_info = env.step(action)[brain_name]        # send the action to the environment
    next_state = env_info.vector_observations[0]   # get the next state
    reward = env_info.rewards[0]                   # get the reward
    done = env_info.local_done[0]                  # see if episode has finished
    score += reward                                # update the score
    state = next_state                             # roll over the state to next time step
    if done:                                       # exit loop if episode finished
        break
    
print(f"Score: {score}")

# when finish testing, close the environment
env.close()