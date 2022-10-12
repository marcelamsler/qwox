import random
import unittest

import numpy as np
from pettingzoo.test import api_test

from env.qwox_env import QwoxEnv
from env.wrapped_quox_env import wrapped_quox_env


class EnvironmentTest(unittest.TestCase):

    def test_env(self):
        env = wrapped_quox_env()
        api_test(env, num_cycles=11, verbose_progress=True)

    # Test does not work, as api_test sends ints as actions instead of
    # proper array based on action space, which causes IllegalActionWrapper to
    # exit, as the int is not part of the unflattened action_mask
    # def test_wrapped_env(self):
    #    env = wrapped_quox_env()
    #    api_test(env, num_cycles=20, verbose_progress=True)

    def test_get_round(self):
        first_round = QwoxEnv.get_round(total_started_step_count=4, agent_count=4)
        self.assertEqual(1, first_round)

        after_last_action_in_round = QwoxEnv.get_round(total_started_step_count=8, agent_count=4)
        self.assertEqual(1, after_last_action_in_round)

        after_last_action_in_round_with_two_players = QwoxEnv.get_round(total_started_step_count=3, agent_count=2)
        self.assertEqual(1, after_last_action_in_round_with_two_players)

        some_round = QwoxEnv.get_round(total_started_step_count=6, agent_count=2)
        self.assertEqual(2, some_round)

        some_round = QwoxEnv.get_round(total_started_step_count=6, agent_count=2)
        self.assertEqual(2, some_round)

        some_round = QwoxEnv.get_round(total_started_step_count=8, agent_count=2)
        self.assertEqual(2, some_round)

    def test_is_second_part_of_round(self):
        after_two_actions = QwoxEnv.is_second_part_of_round(total_started_step_count=3, num_agents=2)
        self.assertEqual(True, after_two_actions)

        after_one_action_in_second_round = QwoxEnv.is_second_part_of_round(total_started_step_count=5, num_agents=2)
        self.assertEqual(False, after_one_action_in_second_round)

    def test_environment_manually(self):
        env = wrapped_quox_env()
        env.reset()
        for agent in env.agent_iter():
            observation, reward, done, info = env.last()
            print("last reward for this agent", agent, reward, info)

            if np.count_nonzero(observation["action_mask"][:44]) > 0:
                action = random.choice(np.flatnonzero(observation["action_mask"]))
            else:
                print("Couldn't find possible action, so taking 54")
                action = 54

            env.step(action)


if __name__ == '__main__':
    unittest.main()
