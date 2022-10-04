import unittest

import numpy as np
from numpy import int8
from numpy.testing import assert_array_equal

from game_models.color import Color
from game_models.dice import Dice
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

    def test_valid_actions_with_one_dices_first_part_of_round(self):
        card = GameCard("some_player_id")
        allowed_actions = card.get_allowed_actions_mask(dices=GameCardTest.get_dices_with_value(value=1),
                                                        is_tossing_player=True, part_of_round=1)

        expected_action_mask = np.zeros(shape=(4, 11), dtype=int8)
        combined_value_of_white_dices = 2
        expected_action_mask[0][combined_value_of_white_dices - 2] = 1
        expected_action_mask[1][combined_value_of_white_dices - 2] = 1
        expected_action_mask[2][12 - combined_value_of_white_dices] = 1
        expected_action_mask[3][12 - combined_value_of_white_dices] = 1

        assert_array_equal(allowed_actions, expected_action_mask)

    def test_valid_actions_with_one_dices_second_part_of_round(self):
        card = GameCard("some_player_id")
        allowed_actions = card.get_allowed_actions_mask(dices=GameCardTest.get_dices_with_value(value=4),
                                                        is_tossing_player=True, part_of_round=2)

        expected_action_mask = np.zeros(shape=(4, 11), dtype=int8)
        combined_value_of_white_and_one_colored_dice = 8
        expected_action_mask[0][combined_value_of_white_and_one_colored_dice - 2] = 1
        expected_action_mask[1][combined_value_of_white_and_one_colored_dice - 2] = 1
        expected_action_mask[2][12 - combined_value_of_white_and_one_colored_dice] = 1
        expected_action_mask[3][12 - combined_value_of_white_and_one_colored_dice] = 1

        assert_array_equal(allowed_actions, expected_action_mask)

    def test_valid_actions_with_one_dices_second_part_of_round_not_tossing_player(self):
        card = GameCard("some_player_id")
        allowed_actions = card.get_allowed_actions_mask(dices=GameCardTest.get_dices_with_value(),
                                                        is_tossing_player=False, part_of_round=2)

        assert_array_equal(allowed_actions, np.zeros(shape=(4, 11), dtype=int8))

    def test_valid_actions_after_some_steps(self):
        card = GameCard("some_player_id")
        card.cross_value_in_line(Color.RED, 2)
        card.cross_value_in_line(Color.YELLOW, 12)
        card.cross_value_in_line(Color.GREEN, 7)

        expected_action_map = np.zeros(shape=(4, 11), dtype=int8)
        expected_action_map[Color.RED.value][6 - 2] = 1
        expected_action_map[Color.GREEN.value][12 - 6] = 1
        expected_action_map[Color.BLUE.value][12 - 6] = 1

        allowed_actions = card.get_allowed_actions_mask(dices=GameCardTest.get_dices_with_value(3),
                                                        is_tossing_player=False, part_of_round=1)
        assert_array_equal(allowed_actions, expected_action_map)

    def test_mask_for_dices(self):
        dice_value = 3
        mask = GameCard.get_mask_based_on_dices(dices=GameCardTest.get_dices_with_value(dice_value),
                                                is_tossing_player=False,
                                                part_of_round=1)

        expected_action_map = np.zeros(shape=(4, 11), dtype=int8)
        value_of_white_dices = dice_value * 2
        expected_action_map[Color.RED.value][value_of_white_dices - 2] = 1
        expected_action_map[Color.YELLOW.value][value_of_white_dices - 2] = 1
        expected_action_map[Color.GREEN.value][12 - value_of_white_dices] = 1
        expected_action_map[Color.BLUE.value][12 - value_of_white_dices] = 1

        assert_array_equal(mask, expected_action_map)

    @staticmethod
    def get_dices_with_value(value: int = 1):
        return [Dice(color, value) for color in
                [Color.RED, Color.YELLOW, Color.GREEN, Color.BLUE, Color.WHITE, Color.WHITE]]


if __name__ == '__main__':
    unittest.main()
