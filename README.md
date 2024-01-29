# Bomberland engine + starter kits

## About

[Bomberland](https://www.gocoder.one/bomberland) is a multi-agent AI competition inspired by the classic console game Bomberman.

Teams build intelligent agents using strategies from tree search to deep reinforcement learning. The goal is to compete in a 2D grid world collecting power-ups and placing explosives to take your opponent down.

This repo contains starter kits for working with the game engine + the engine source!

![Bomberland multi-agent environment](./engine/bomberland-ui/src/source-filesystem/docs/2-environment-overview/bomberland-preview.gif "Bomberland")

## Usage

### Prerequisites

- Python: 3.8.15
- Docker: 20.10.23+
- Docker-Compose: 2.15.1+
- PyTorch: 2.1.2

Important note: if you are Mac M1/M2 user, please checkout to [this](https://github.com/danorel/bomberland/tree/mac-m1-or-m2) branch.
 
### Basic usage

See: [Documentation](https://www.gocoder.one/docs)

1. Clone or download this repo.
2. To train agents, run from the root directory:

a. Train DQN:

```shell
docker-compose -f docker-compose.train.yaml up gym agent-dqn-training --force-recreate --abort-on-container-exit
```

b. Train PPO:

```shell
docker-compose -f docker-compose.train.yaml up gym agent-ppo-training --force-recreate --abort-on-container-exit
```

3. To test the effeciency of the agents on the client, run from the root directory:

```shell
docker-compose -f docker-compose.test.yaml up --force-recreate --abort-on-container-exit
```

While the command is running and agents are preparing to play the game, access the client by going to `http://localhost:3000/` in your browser.

4. From the client, you can connect as a `spectator` or `agent` (to play as a human player)


5. To submit the agent for competition:

a. Copy and save a .pt file with your model.

b. Send the .pt file to [Danyil Orel](https://t.me/danorel).
