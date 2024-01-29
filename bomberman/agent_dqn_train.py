import asyncio
import datetime
import math
import random
import time
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from components.environment.config import (
    ACTIONS,
    FWD_MODEL_CONNECTION_DELAY,
    FWD_MODEL_CONNECTION_RETRIES, 
    FWD_MODEL_URI, 
)
from components.environment.gym import Gym, GymEnv
from components.environment.mocks import MOCK_15x15_INITIAL_OBSERVATION
from components.models.dqn import DQNAgent, ReplayMemory, Transition
from components.action import make_action
from components.reward import calculate_reward
from components.state import (
    action_dimensions, 
    state_dimensions, 
    observation_to_state
)
from components.types import State
from components.utils.device import device

"""
Simulation of two agents playing one againts the other.
"""
AGENTS = ['a', 'b']
UNITS = ["c", "d", "e", "f", "g", "h"]

"""
Hyperparameters
"""

EPOCHS = 5
STEPS = 1000
BATCH_SIZE = 128
LEARNING_RATE = 0.0003
GAMMA = 0.99
TAU = 0.005
EPS_MIN = 0.05
EPS_MAX = 0.3
EPS_DECAY = 1000
PRINT_EVERY = 100

"""
Epsilon-greedy action selection.
"""
def select_action(agent: DQNAgent, state: State, steps_done: int, verbose: bool = True):

    agent_id = AGENTS[steps_done % 2]
    unit_id = UNITS[steps_done % 6]
    
    if verbose:
        print(f"Agent: {agent_id}, Unit: {unit_id}")

    eps_threshold = EPS_MIN + (EPS_MAX - EPS_MIN) * \
        math.exp(-1. * steps_done / EPS_DECAY)

    if random.random() <= eps_threshold:
        action = random.randrange(len(ACTIONS))
    else:
        with torch.no_grad():
            probs = agent(state)
            action = torch.argmax(probs)

    action = torch.tensor(action, dtype=torch.int64).reshape(1)

    return action, (agent_id, unit_id)

"""
Optimize memory samples.
"""
def optimize_model(policy_net: DQNAgent, target_net: DQNAgent, optimizer, memory: ReplayMemory):
    if len(memory) < BATCH_SIZE:
        return

    transitions = memory.sample(BATCH_SIZE)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.stack([s for s in batch.next_state
                                                 if s is not None])
    state_batch = torch.stack(batch.state)
    action_batch = torch.stack(batch.action)
    reward_batch = torch.stack(batch.reward)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1).values
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    with torch.no_grad():
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1).values
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    # In-place gradient clipping
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()


async def train(env: GymEnv, policy_net: DQNAgent, target_net: DQNAgent, optimizer, memory: ReplayMemory):
    cumulative_rewards = []

    for epoch in range(EPOCHS):
        print(f"Started {epoch} epoch...")
        cumulative_reward = 0

        # Initialize the environment and get it's state
        prev_observation = await env.reset()
        prev_state = observation_to_state(prev_observation, current_agent_id='a', current_unit_id='c')

        # Iterate and gather experience
        for steps_done in range(STEPS):
            action, (agent_id, unit_id) = select_action(policy_net, prev_state, steps_done)
            action_or_idle = make_action(prev_observation, agent_id, unit_id, action=int(action.item()))
            action_is_idle = action_or_idle is None

            if action_is_idle:
                next_observation, done, info = await env.step([])
            else:
                next_observation, done, info = await env.step([action_or_idle])

            reward = calculate_reward(prev_observation, next_observation, current_agent_id=agent_id, current_unit_id=unit_id)
            next_state = observation_to_state(next_observation, current_agent_id=agent_id, current_unit_id=unit_id)

            # Store the transition in memory
            memory.push(prev_state, action, next_state, reward)

            # Perform one step of the optimization (on the policy network)
            optimize_model(policy_net, target_net, optimizer, memory)

            # Soft update of the target network's weights
            # θ′ ← τ θ + (1 −τ )θ′
            target_net_state_dict = target_net.state_dict()
            policy_net_state_dict = policy_net.state_dict()
            for key in policy_net_state_dict:
                target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
            target_net.load_state_dict(target_net_state_dict)

            prev_state = next_state
            prev_observation = next_observation

            # Compute statistics
            cumulative_reward += reward.item()

            if steps_done % PRINT_EVERY == 0:
                print(f"Action: {action}")
                print(f"Reward: {reward}, Done: {done}, Info: {info}, Observation: {next_observation}")

            if done:
                print(f"Agent achieved the goal during step {steps_done}")
                break

        # Compute statistics
        cumulative_rewards.append(cumulative_reward)
    
    print("Drawing plot: reward distribution over epochs")
    epochs = range(1, EPOCHS + 1) 
    ax = plt.axes()
    ax.plot(epochs, cumulative_rewards)
    ax.set_title('Cumulative reward by epoch')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Cumulative reward')
    ax.xaxis.set_ticks(epochs)
    plt.savefig("agent_dqn_rewards.png")


async def main():
    print("============================================================================================")
    print("DQN agent")
    print("Connecting to gym")
    gym = Gym(FWD_MODEL_URI)
    for retry in range(1, FWD_MODEL_CONNECTION_RETRIES):
        try:
            await gym.connect()
        except:
            print(f"Retrying to connect with {retry} attempt...")
            time.sleep(FWD_MODEL_CONNECTION_DELAY)
            continue
        break
    print("Connected to gym successfully")
    print("============================================================================================")

    print("============================================================================================")
    print("Initializing agent")
    env = gym.make("bomberland-gym", MOCK_15x15_INITIAL_OBSERVATION)
    observation = await env.reset()
    n_states = state_dimensions(observation)
    n_actions = action_dimensions()
    print(f"Agent: states = {n_states}, actions = {n_actions}")

    policy_net = DQNAgent(n_states, n_actions)
    target_net = DQNAgent(n_states, n_actions)
    target_net.load_state_dict(policy_net.state_dict())

    optimizer = optim.AdamW(policy_net.parameters(), lr=LEARNING_RATE, amsgrad=True)
    memory = ReplayMemory(8192)
    print("============================================================================================")

    print("============================================================================================")
    print("Training agent")
    start_time = datetime.datetime.now().replace(microsecond=0)
    print("Started training at (GMT) : ", start_time)
    await train(env, policy_net, target_net, optimizer, memory)
    end_time = datetime.datetime.now().replace(microsecond=0)
    print("Started training at (GMT) : ", start_time)
    print("Finished training at (GMT) : ", end_time)
    print("Total training time  : ", end_time - start_time)
    print("============================================================================================")

    print("============================================================================================")
    print("Saving agent")
    policy_net.save()
    policy_net.show()
    print("============================================================================================")
    
    await gym.close()


if __name__ == "__main__":
    asyncio.run(main())
