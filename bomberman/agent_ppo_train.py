import asyncio
import datetime
import time
import matplotlib.pyplot as plt

from components.environment.config import (
    FWD_MODEL_CONNECTION_DELAY,
    FWD_MODEL_CONNECTION_RETRIES, 
    FWD_MODEL_URI,  
)
from components.environment.gym import Gym, GymEnv
from components.environment.observation import observation_from_state
from components.environment.mocks import MOCK_15x15_INITIAL_OBSERVATION
from components.models.ppo import PPO
from components.action import make_action
from components.reward import calculate_reward
from components.state import (
    action_dimensions, 
    state_dimensions, 
    # component_to_state
)
from components.types import State

"""
Simulation of two agents playing one againts the other.
"""
AGENTS = ['a', 'b']
UNITS = ["c", "d", "e", "f", "g", "h"]

"""
Hyperparameters
"""

EPOCHS = 500
STEPS = 10000
BATCH_SIZE = 128
LEARNING_RATE_ACTOR = 0.0003
LEARNING_RATE_CRITIC = 0.001
K_EPOCHS = 80  # update policy for K epochs in one PPO update
GAMMA = 0.99
TAU = 0.005
EPS_CLIP = 0.2  # clip parameter for PPO
ACTION_STD = 0.6
HAS_CONTINUOUS_ACTION_SPACE = False
PRINT_EVERY = 100
UPDATE_EVERY = 100
SAVE_EVERY = 10000

"""
Epsilon-greedy action selection.
"""
def select_action(agent: PPO, state: State, steps_done: int, verbose: bool = True):

    agent_id = AGENTS[steps_done % 2]
    unit_id = UNITS[steps_done % 6]
    
    if verbose:
        print(f"Agent: {agent_id}, Unit: {unit_id}")
    action = agent.select_action(state)

    return action, (agent_id, unit_id)


async def train(env: GymEnv, agent: PPO):
    cumulative_rewards = []
    survival_times = []  # Додавання для збору інформації про час виживання

    for epoch in range(EPOCHS):
        print(f"Started {epoch} epoch...")
        cumulative_reward = 0
        survival_time = 0  # Ініціалізація лічильника часу виживання

        # Initialize the environment and get it's state
        prev_observation = await env.reset()
        prev_state = observation_from_state(prev_observation, my_unit_id='c')

        # Iterate and gather experience
        for steps_done in range(1, STEPS):
            action, (agent_id, unit_id) = select_action(agent, prev_state, steps_done)
            print("after")
            action_or_idle = make_action(prev_observation, agent_id, unit_id, action)
            action_is_idle = action_or_idle is None

            if action_is_idle:
                next_observation, done, info = await env.step([])
            else:
                next_observation, done, info = await env.step([action_or_idle])

            reward = calculate_reward(prev_observation, next_observation, current_agent_id=agent_id, current_unit_id=unit_id)
            next_state = observation_from_state(next_observation, my_unit_id=unit_id)

            # saving reward and is_terminals
            agent.buffer.rewards.append(reward)
            agent.buffer.is_terminals.append(done)

            prev_state = next_state
            prev_observation = next_observation

            # Compute statistics
            cumulative_reward += reward.item()
            survival_time += 1  # Оновлення лічильника часу виживання

            if steps_done % UPDATE_EVERY == 0:
                agent.update()

            if steps_done % PRINT_EVERY == 0:
                print(f"Action: {action}")
                print(f"Reward: {reward}, Done: {done}, Info: {info}, Observation: {next_observation}")

            if done:
                print(f"Agent achieved the goal during step {steps_done}")
                break

        # Compute statistics
        cumulative_rewards.append(cumulative_reward)
        survival_times.append(survival_time)  # Збереження часу виживання за епоху
    
    print("Drawing plot: reward distribution over epochs")
    epochs = range(1, EPOCHS + 1) 
    ax = plt.axes()
    ax.plot(epochs, cumulative_rewards)
    ax.set_title('Cumulative reward by epoch')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Cumulative reward')
    ax.xaxis.set_ticks(epochs)
    plt.savefig("agent_ppo_rewards.png")

    # Plotting Average Survival Time
    plt.plot(epochs, survival_times)
    plt.title('Average Survival Time by epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Survival Time (steps)')
    plt.tight_layout()
    plt.savefig("agent_ppo_training_metrics.png")

    def plot_metrics(epochs, cumulative_rewards, survival_times):
        fig, ax1 = plt.subplots(figsize=(10, 6))

        color = 'tab:red'
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Cumulative Reward', color=color)
        ax1.plot(epochs, cumulative_rewards, color=color)
        ax1.tick_params(axis='y', labelcolor=color)

        ax2 = ax1.twinx()  # Створює другу ось Y, яка поділяє ось X з ax1
        color = 'tab:blue'
        ax2.set_ylabel('Survival Time (steps)', color=color)  # Ми вже маємо мітку для осі X
        ax2.plot(epochs, survival_times, color=color)
        ax2.tick_params(axis='y', labelcolor=color)

        fig.tight_layout()  # Щоб запобігти нашаруванню і забезпечити чіткість міток
        plt.title('Cumulative Reward and Survival Time by Epoch')
        plt.savefig("combined_metrics.png")
        plt.show()

    # Припускаючи, що ви вже маєте epochs, cumulative_rewards, і survival_times
    plot_metrics(epochs, cumulative_rewards, survival_times)


async def main():
    print("============================================================================================")
    print("PPO agent")
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
    state = await env.reset()
    observation = observation_from_state(state, 'c')
    print(f"Observation Shape: {observation.shape} \n")
    n_states = observation.shape[0]
    n_actions = action_dimensions()
    print(f"Agent: states = {n_states}, actions = {n_actions}")

    ppo_agent = PPO(
        n_states, 
        n_actions, 
        LEARNING_RATE_ACTOR, 
        LEARNING_RATE_CRITIC,
        GAMMA,
        K_EPOCHS, 
        EPS_CLIP,
        HAS_CONTINUOUS_ACTION_SPACE, 
        ACTION_STD
    )
    print("============================================================================================")

    print("============================================================================================")
    print("Training agent")
    start_time = datetime.datetime.now().replace(microsecond=0)
    print("Started training at (GMT) : ", start_time)
    await train(env, ppo_agent)
    end_time = datetime.datetime.now().replace(microsecond=0)
    print("Started training at (GMT) : ", start_time)
    print("Finished training at (GMT) : ", end_time)
    print("Total training time  : ", end_time - start_time)
    print("============================================================================================")

    print("============================================================================================")
    print("Saving agent")
    ppo_agent.save()
    ppo_agent.show()
    print("============================================================================================")
    
    await gym.close()


if __name__ == "__main__":
    asyncio.run(main())
