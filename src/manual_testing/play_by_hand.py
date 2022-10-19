import os

import numpy as np
import numpy.ma as ma
import torch
from numpy import int8
from tianshou.data import Batch
from tianshou.policy import DQNPolicy
from tianshou.utils.net.common import Net

from env.wrapped_quox_env import wrapped_quox_env


def get_trained_agent(env):
    observation_space = (

    )

    net = Net(
        state_shape=env.observation_spaces[env.agents[0]]["observation"].shape,
        action_shape=env.observation_spaces[env.agents[0]]["action_mask"].shape,
        hidden_sizes=[128, 128, 128, 128, 128, 128],
        device="cuda" if torch.cuda.is_available() else "cpu",
    ).to("cuda" if torch.cuda.is_available() else "cpu")

    optim = torch.optim.Adam(net.parameters(), lr=1e-3)

    agent_learn = DQNPolicy(
        model=net,
        optim=optim,
        discount_factor=0.9,
        estimation_step=20,
        target_update_freq=320,
    )

    return agent_learn


if __name__ == "__main__":
    env = wrapped_quox_env()
    env.reset()

    path = os.path.join("policy-69.pth")
    trained_agent = get_trained_agent(env)
    trained_agent.load_state_dict(torch.load(path))

    for agent in env.agent_iter():
        observation, reward, _, info = env.last()

        if agent == "player_1":
            env.render()
            print("Possible actions:")
            action_mask = ma.array(np.arange(0, 55, dtype=int8).reshape(5, 11), mask=np.logical_not(observation["action_mask"]).astype(int))
            print(action_mask)
            var = input("Choose your action: ")
            action = int(var)
            print("You entered: " + var)
        else:
            action = trained_agent(Batch(obs=Batch(obs=np.array([observation["observation"]]), mask=np.array([observation["action_mask"]])), info=info)).act[0]
            print("Opponent chose: ", action)


        env.step(action)

        if env.unwrapped.board.game_is_finished():
            print("game finished")
            break
