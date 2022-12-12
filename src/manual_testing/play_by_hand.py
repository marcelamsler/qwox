import os

import numpy as np
import torch
from tianshou.data import Batch
from tianshou.policy import DQNPolicy
from tianshou.utils.net.common import Net

from env.wrapped_quox_env import wrapped_quox_env
from manual_testing.utils import get_trained_agent, create_batch_from_observation

if __name__ == "__main__":
    env = wrapped_quox_env()
    env.reset()

    trained_agent = get_trained_agent(env, 103)

    for agent in env.agent_iter():
        observation, reward, _, info = env.last()

        if agent == "player_1":
            env.unwrapped.render()
            print("Possible actions:")
            obs = observation["observation"][0].reshape(5, 12)[:5, :11].flatten()
            obs = np.where(obs == 1, ["X "], ["--"])

            action_mask = np.where(observation["action_mask"] == 1, np.arange(0, 55, dtype=int), -1)
            readable_action_mask = np.where(action_mask == -1, obs, action_mask)
            print(readable_action_mask.reshape(5, 11))
            var = input("Choose your action: ")
            action = int(var)
            print("You entered: " + var)
        else:
            action = trained_agent(create_batch_from_observation(observation, info)).act[0]
            print("Opponent chose: ", action)

        env.step(action)

        if env.unwrapped.board.is_game_finished():
            print("game finished")
            break
