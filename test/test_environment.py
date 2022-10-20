import copy
import random
import unittest

import numpy as np
from numpy.testing import assert_array_equal
from pettingzoo.test import api_test

from env.qwox_env import QwoxEnv
from env.wrapped_quox_env import wrapped_quox_env
from utils import get_dices_with_value


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
        after_two_actions = QwoxEnv.is_second_part_of_round(total_started_step_count=2, num_agents=2)
        self.assertEqual(False, after_two_actions)

        after_three_actions = QwoxEnv.is_second_part_of_round(total_started_step_count=3, num_agents=2)
        self.assertEqual(True, after_three_actions)

        after_one_action_in_second_round = QwoxEnv.is_second_part_of_round(total_started_step_count=5, num_agents=2)
        self.assertEqual(False, after_one_action_in_second_round)

        after_3_actions_in_second_round = QwoxEnv.is_second_part_of_round(total_started_step_count=8, num_agents=2)
        self.assertEqual(True, after_3_actions_in_second_round)

    def test_environment_manually(self):
        env = wrapped_quox_env()
        env.reset()
        for agent in env.agent_iter():
            observation, reward, _, info = env.last()

            action = random.choice(np.flatnonzero(observation["action_mask"]))

            env.render()
            env.step(action)

            if env.unwrapped.board.game_is_finished():
                return

    def test_get_round(self):
        round = QwoxEnv.get_round(total_started_step_count=1, agent_count=2)
        self.assertEqual(1, round)

        round = QwoxEnv.get_round(total_started_step_count=5, agent_count=2)
        self.assertEqual(2, round)

        round = QwoxEnv.get_round(total_started_step_count=3, agent_count=2)
        self.assertEqual(1, round)

        round = QwoxEnv.get_round(total_started_step_count=4, agent_count=2)
        self.assertEqual(1, round)

    def test_environment_through_multiple_steps(self):
        env = wrapped_quox_env()
        env.reset()
        env_uw: QwoxEnv = env.unwrapped
        env_uw.board.dices = get_dices_with_value(value=1)
        dices_round_1 = copy.deepcopy(env_uw.board.dices[:])

        # agent1 step 1 round 1
        self.assertEqual(1, env_uw.current_round)
        observation, reward, _, info = env.last()
        first_action_of_player_1 = random.choice(np.flatnonzero(observation["action_mask"]))
        self.assertEqual(False, env_uw.board.game_cards["player_1"].crossed_something_in_current_round)
        env.step(first_action_of_player_1)

        # agent2 step 1 round 1
        observation, reward, _, info = env.last()
        action = random.choice(np.flatnonzero(observation["action_mask"]))
        self.assertEqual(2, env_uw.total_started_step_count)
        self.assertEqual(False, env_uw.board.game_cards["player_2"].crossed_something_in_current_round)
        env.step(action)

        # agent1 step 2 round 1
        observation, reward, _, info = env.last()
        action = random.choice(np.flatnonzero(observation["action_mask"]))
        self.assertEqual(3, env_uw.total_started_step_count)
        self.assertEqual(0, env_uw.get_tossing_agent_index(current_round=env_uw.current_round))
        if first_action_of_player_1 > 47:
            self.assert_contains_passing_fields(observation["action_mask"])

        env.step(action)

        self.assertEqual(False, env_uw.board.game_cards["player_1"].crossed_something_in_current_round)

        # agent2 step 2 round 1
        observation, reward, _, info = env.last()
        self.assertEqual(1, env_uw.current_round)
        self.assertEqual(0, env_uw.get_tossing_agent_index(current_round=env_uw.current_round))
        assert_array_equal(env_uw.board.dices, dices_round_1)
        self.assertEqual(4, env_uw.total_started_step_count)

        action = random.choice(np.flatnonzero(observation["action_mask"]))
        env.step(action)

        self.assertEqual(False, env_uw.board.game_cards["player_2"].crossed_something_in_current_round)
        if np.array_equal(env_uw.board.dices, dices_round_1):
            print("Dices should not have the same value for the next round")
            self.assertFalse(True)

        #### Round 2

        # agent1 step 1 round 2
        observation, reward, _, info = env.last()
        action = random.choice(np.flatnonzero(observation["action_mask"]))
        self.assertEqual(5, env_uw.total_started_step_count)
        self.assertEqual(2, env_uw.current_round)
        self.assertEqual(1, env_uw.get_tossing_agent_index(current_round=env_uw.current_round))
        self.assertFalse(env_uw.board.game_cards["player_1"].crossed_something_in_current_round)
        env.step(action)

        # agent2 step 1 round 2
        observation, reward, _, info = env.last()
        first_action_of_player2_in_second_round = random.choice(np.flatnonzero(observation["action_mask"]))
        self.assertFalse(env_uw.board.game_cards["player_2"].crossed_something_in_current_round)
        env.step(first_action_of_player2_in_second_round)
        if first_action_of_player2_in_second_round > 47:
            self.assertFalse(env_uw.board.game_cards["player_2"].crossed_something_in_current_round)

        # agent1 step 2 round 2
        observation, reward, _, info = env.last()
        action = random.choice(np.flatnonzero(observation["action_mask"]))
        self.assertEqual(1, env_uw.get_tossing_agent_index(current_round=env_uw.current_round))
        env.step(action)

        # agent2 step 2 round 2
        observation, reward, _, info = env.last()
        action = random.choice(np.flatnonzero(observation["action_mask"]))

        if first_action_of_player2_in_second_round > 47:
            self.assertFalse(env_uw.board.game_cards["player_2"].crossed_something_in_current_round)
            self.assert_contains_passing_fields(observation["action_mask"])
        self.assertEqual(1, env_uw.get_tossing_agent_index(current_round=env_uw.current_round))
        env.step(action)

        # Round 3
        self.assertEqual(3, env_uw.current_round)

    def assert_contains_passing_fields(self, action_mask):
        self.assertEqual(1, action_mask[44])
        self.assertEqual(1, action_mask[45])
        self.assertEqual(1, action_mask[46])
        self.assertEqual(1, action_mask[47])


if __name__ == '__main__':
    unittest.main()
