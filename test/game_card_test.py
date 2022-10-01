import unittest

import numpy as np
from numpy import int8
from numpy.testing import assert_array_equal

from game_models.color import Color
from game_models.game_card import GameCard


class GameCardTest(unittest.TestCase):
    def test_crossing_numbers(self):
        card = GameCard("some_player_id")
        card.cross_value_in_line(Color.RED, 4)
        card.cross_value_in_line(Color.BLUE, 2)
        expected_state = np.zeros(shape=(4, 11), dtype=int8)
        expected_state[Color.RED.value][2] = 1
        expected_state[Color.BLUE.value][10] = 1
        assert_array_equal(card.get_state(), expected_state)

    def test_cross_value_with_action(self):
        card = GameCard("some_player_id")
        action = np.zeros(shape=(4, 11), dtype=int8)
        action[Color.RED.value][2] = 1
        card.cross_value_with_action(action)

        expected_state = np.zeros(shape=(4, 11), dtype=int8)
        expected_state[Color.RED.value][2] = 1
        assert_array_equal(card.get_state(), expected_state)

    def test_cross_value_with_action_with_start_state(self):
        card = GameCard("some_player_id")
        action = np.zeros(shape=(4, 11), dtype=int8)
        card.cross_value_in_line(Color.BLUE, 2)
        action[Color.YELLOW.value][5] = 1
        card.cross_value_with_action(action)

        expected_state = np.zeros(shape=(4, 11), dtype=int8)
        expected_state[Color.YELLOW.value][5] = 1
        expected_state[Color.BLUE.value][10] = 1
        assert_array_equal(card.get_state(), expected_state)

    def test_valid_actions_at_beginning(self):
        card = GameCard("some_player_id")
        allowed_actions = card.get_allowed_actions_mask()
        assert_array_equal(allowed_actions, np.ones(shape=(4, 11), dtype=int8))

    def test_valid_actions_after_some_steps(self):
        card = GameCard("some_player_id")
        card.cross_value_in_line(Color.YELLOW, 12)
        card.cross_value_in_line(Color.GREEN, 7)
        card.cross_value_in_line(Color.RED, 2)


        expected_action_map = np.ones(shape=(4, 11), dtype=int8)
        expected_action_map[Color.GREEN.value][0:6] = 0
        expected_action_map[Color.YELLOW.value][:] = 0
        expected_action_map[Color.RED.value][:0 + 1] = 0
        allowed_actions = card.get_allowed_actions_mask()
        assert_array_equal(allowed_actions, expected_action_map)


if __name__ == '__main__':
    unittest.main()
