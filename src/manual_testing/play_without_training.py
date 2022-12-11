import torch
from tianshou.policy import RandomPolicy

from agents.tianshou.long_playing_policy import LongPlayingPolicy
from env.wrapped_quox_env import wrapped_quox_env
from manual_testing.utils import get_trained_agent, create_batch_from_observation, \
    create_batch_from_observation_for_handmade_policy

if __name__ == "__main__":
    env = wrapped_quox_env()
    env.reset()

    agent1 = get_trained_agent(env, 103)
    agent2 = RandomPolicy()

    with torch.no_grad():
        for _ in range(500):
            env.reset()
            for agent in env.agent_iter():
                observation, reward, _, info = env.last()

                if agent == "player_1":
                    action = agent1(create_batch_from_observation(observation, info)).act[0]
                else:
                    action = agent2(create_batch_from_observation(observation, info)).act[0]

                env.step(action)

                if env.unwrapped.board.is_game_finished():
                    print("game finished")
                    break
