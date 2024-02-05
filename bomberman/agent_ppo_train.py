import torch
import torch.nn as nn
import numpy as np
import asyncio
import datetime
import time
import matplotlib.pyplot as plt
import optuna
from stable_baselines3.common.callbacks import EvalCallback
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from optuna.visualization import plot_optimization_history, plot_param_importances
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
N_TRIALS = 100  # Maximum number of trials
N_JOBS = 1 # Number of jobs to run in parallel
N_STARTUP_TRIALS = 10  # Stop random sampling after N_STARTUP_TRIALS
N_EVALUATIONS = 3  # Number of evaluations during the training
N_TIMESTEPS = int(2e3)  # Training budget
EVAL_FREQ = int(N_TIMESTEPS / N_EVALUATIONS)
N_EVAL_ENVS = 5
N_EVAL_EPISODES = 10
TIMEOUT = int(60 * 60)  # 15 minutes

EPOCHS = 500
BATCH_SIZE = 128
DEFAULT_HYPERPARAMS = {
"STEPS": 10000,
"LEARNING_RATE_ACTOR": 0.0003,
"LEARNING_RATE_CRITIC": 0.001,
"K_EPOCHS": 80,  # update policy for K epochs in one PPO update
"GAMMA": 0.99,
"TAU": 0.005,
"EPS_CLIP": 0.2,  # clip parameter for PPO
"ACTION_STD": 0.6,
"ACTIVATION_FN": nn.Tahn
}
HAS_CONTINUOUS_ACTION_SPACE = False
PRINT_EVERY = 100
UPDATE_EVERY = 100
SAVE_EVERY = 10000

def softmax_action_selection(agent: PPO, state: State, steps_done: int, observation):
    agent_id = AGENTS[steps_done % 2]
    unit_id = UNITS[steps_done % 6]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Перетворення стану на тензор PyTorch і використання моделі для отримання ймовірностей дій
    state_tensor = torch.FloatTensor(state).to(device)
    with torch.no_grad():
        action_probs = agent.policy_old.actor(state_tensor)

    # Використання np.random.choice для вибору дії на основі отриманих ймовірностей
    action = np.random.choice(np.arange(action_probs.size(-1)), p=action_probs.numpy())

    # Виклик make_action з отриманою дією
    action_packet = make_action(observation, agent_id, unit_id, action)

    return action_packet

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

def sample_ppo_params(trial: optuna.Trial):
    BATCH_SIZE = trial.suggest_categorical("BATCH_SIZE", [8, 16, 32, 64, 128, 256, 512])
    STEPS = 2 ** trial.suggest_int("STEPS", 3, 12)
    LEARNING_RATE_ACTOR = trial.suggest_float("LEARNING_RATE_ACTOR", 1e-5, 1.0)
    LEARNING_RATE_CRITIC = trial.suggest_float("LEARNING_RATE_CRITIC", 1e-3, 1.0),
    GAMMA = 1.0 - trial.suggest_float("GAMMA", 0.0001, 0.1, log=True)
    TAU =  trial.suggest_float("TAU", 1e-5, 0.1),
    EPS_CLIP = trial.suggest_float("EPS_CLIP", 0.1, 0.4),  # clip parameter for PPO
    NET_ARCH_TYPE = trial.suggest_categorical("net_arch", ["tiny", "small", "medium"])
    ACTION_STD = trial.suggest_float("ACTION_STD", 0.01, 0.99)

    activation_fn = {"tanh": nn.Tanh, "relu": nn.ReLU, "elu": nn.ELU, "leaky_relu": nn.LeakyReLU}[ACTIVATION_FN]


class TrialEvalCallback(EvalCallback):
    """
    Callback used for evaluating and reporting a trial.

    :param eval_env: Evaluation environement
    :param trial: Optuna trial object
    :param n_eval_episodes: Number of evaluation episodes
    :param eval_freq:   Evaluate the agent every ``eval_freq`` call of the callback.
    :param deterministic: Whether the evaluation should
        use a stochastic or deterministic policy.
    :param verbose:
    """

    def __init__(
            self,
            eval_env: gym.Env,
            trial: optuna.Trial,
            n_eval_episodes: int = 5,
            eval_freq: int = 10000,
            deterministic: bool = True,
            verbose: int = 0,
    ):

        super().__init__(
            eval_env=eval_env,
            n_eval_episodes=n_eval_episodes,
            eval_freq=eval_freq,
            deterministic=deterministic,
            verbose=verbose,
        )
        self.trial = trial
        self.eval_idx = 0
        self.is_pruned = False

    def _on_step(self) -> bool:
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            # Evaluate policy (done in the parent class)
            super()._on_step()
            self.eval_idx += 1
            # Send report to Optuna
            self.trial.report(self.last_mean_reward, self.eval_idx)
            # Prune trial if need
            if self.trial.should_prune():
                self.is_pruned = True
                return False
        return True


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

            reward = calculate_reward(prev_observation, next_observation, current_agent_id=agent_id,
                                      current_unit_id=unit_id, action_is_idle=action_is_idle)
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
