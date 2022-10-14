import unittest

import numpy as np
from numpy import int8
from numpy.testing import assert_array_equal

from game_models.color import Color
from game_models.game_card import GameCard
from utils import get_dices_with_value


class GameCardTest(unittest.TestCase):

    def test_crossing_numbers(self):
        card = GameCard("some_player_id")
        card._cross_value_in_line(Color.RED, 4)
        card._cross_value_in_line(Color.BLUE, 2)
        expected_state = np.zeros(shape=GameCard.OBSERVATION_SHAPE, dtype=int8)
        expected_state[Color.RED.value][2] = 1
        expected_state[Color.BLUE.value][10] = 1
        assert_array_equal(card.get_state(), expected_state)

    def test_cross_value_with_flat_index_action(self):
        card = GameCard("some_player_id")
        card.cross_value_with_flattened_action(2)

        expected_state = np.zeros(shape=GameCard.OBSERVATION_SHAPE, dtype=int8)
        expected_state[Color.RED.value][2] = 1
        assert_array_equal(card.get_state(), expected_state)

    def test_row_closing(self):
        card = GameCard("some_player_id")

        red_row = card._rows[0]
        red_row[0:5] = 1
        card.cross_value_with_flattened_action(10)

        expected_state = np.zeros(shape=GameCard.OBSERVATION_SHAPE, dtype=int8)
        expected_state[Color.RED.value][0:5] = 1
        expected_state[Color.RED.value][10:] = 1
        assert_array_equal(card.get_state(), expected_state)

    def test_row_closing_without_enough_fields_crossed(self):
        card = GameCard("some_player_id")

        red_row = card._rows[0]
        red_row[0:3] = 1
        card.cross_value_with_flattened_action(10)

        expected_state = np.zeros(shape=GameCard.OBSERVATION_SHAPE, dtype=int8)
        expected_state[Color.RED.value][0:3] = 1
        expected_state[Color.RED.value][10] = 1
        expected_state[Color.RED.value][11] = 0
        assert_array_equal(card.get_state(), expected_state)

    def test_cross_value_with_action_with_start_state(self):
        card = GameCard("some_player_id")
        card._cross_value_in_line(Color.BLUE, 2)
        card.cross_value_with_flattened_action(16)

        expected_state = np.zeros(shape=GameCard.OBSERVATION_SHAPE, dtype=int8)
        expected_state[Color.YELLOW.value][5] = 1
        expected_state[Color.BLUE.value][10] = 1
        assert_array_equal(card.get_state(), expected_state)

    def test_valid_actions_with_one_dices_first_part_of_round(self):
        card = GameCard("some_player_id")

        expected_action_mask = np.zeros(shape=GameCard.ACTION_MASK_SHAPE, dtype=int8)
        combined_value_of_white_dices = 2
        expected_action_mask[0][combined_value_of_white_dices - 2] = 1
        expected_action_mask[1][combined_value_of_white_dices - 2] = 1
        expected_action_mask[2][12 - combined_value_of_white_dices] = 1
        expected_action_mask[3][12 - combined_value_of_white_dices] = 1
        expected_action_mask[4][0:3] = 0
        expected_action_mask[4][4:] = 1

        computed_action_mask = card.get_allowed_actions_mask(dices=get_dices_with_value(value=1),
                                                             is_tossing_player=True, is_second_part_of_round=False)

        assert_array_equal(computed_action_mask, expected_action_mask)

    def test_valid_actions_with_value_four_dices_second_part_of_round(self):
        card = GameCard("some_player_id")
        computed_action_mask = card.get_allowed_actions_mask(dices=get_dices_with_value(value=4),
                                                             is_tossing_player=True, is_second_part_of_round=True)

        expected_action_mask = np.zeros(shape=GameCard.ACTION_MASK_SHAPE, dtype=int8)
        combined_value_of_white_and_one_colored_dice = 8
        expected_action_mask[0][combined_value_of_white_and_one_colored_dice - 2] = 1
        expected_action_mask[1][combined_value_of_white_and_one_colored_dice - 2] = 1
        expected_action_mask[2][12 - combined_value_of_white_and_one_colored_dice] = 1
        expected_action_mask[3][12 - combined_value_of_white_and_one_colored_dice] = 1
        self.force_to_use_pass_fields(expected_action_mask)

        assert_array_equal(computed_action_mask, expected_action_mask)

    def test_valid_actions_with_different_value_dices_for_second_part_of_round(self):
        card = GameCard("some_player_id")
        dices = get_dices_with_value(value=4)
        red_dice = [dice for dice in dices if dice.color == Color.RED][0]
        red_dice.current_value = 6
        computed_action_mask = card.get_allowed_actions_mask(dices=dices,
                                                             is_tossing_player=True, is_second_part_of_round=True)

        expected_action_mask = np.zeros(shape=GameCard.ACTION_MASK_SHAPE, dtype=int8)
        combined_value_of_white_and_one_colored_dice = 8
        white_dice_value = 4
        expected_action_mask[0][red_dice.current_value + white_dice_value - 2] = 1
        expected_action_mask[1][combined_value_of_white_and_one_colored_dice - 2] = 1
        expected_action_mask[2][12 - combined_value_of_white_and_one_colored_dice] = 1
        expected_action_mask[3][12 - combined_value_of_white_and_one_colored_dice] = 1
        self.force_to_use_pass_fields(expected_action_mask)

        assert_array_equal(computed_action_mask, expected_action_mask)

    def test_valid_actions_with_different_white_values_for_second_part_of_round(self):
        card = GameCard("some_player_id")
        dices = get_dices_with_value(value=4)
        white_dice1 = [dice for dice in dices if dice.color == Color.WHITE][0]
        white_dice1.current_value = 1
        computed_action_mask = card.get_allowed_actions_mask(dices=dices,
                                                             is_tossing_player=True, is_second_part_of_round=True)

        expected_action_mask = np.zeros(shape=GameCard.ACTION_MASK_SHAPE, dtype=int8)
        combined_value_of_white1_and_one_colored_dice = 5
        combined_value_of_white2_and_one_colored_dice = 8

        # White Dice 1
        expected_action_mask[0][combined_value_of_white1_and_one_colored_dice - 2] = 1
        expected_action_mask[1][combined_value_of_white1_and_one_colored_dice - 2] = 1
        expected_action_mask[2][12 - combined_value_of_white1_and_one_colored_dice] = 1
        expected_action_mask[3][12 - combined_value_of_white1_and_one_colored_dice] = 1

        # White Dice 2
        expected_action_mask[0][combined_value_of_white2_and_one_colored_dice - 2] = 1
        expected_action_mask[1][combined_value_of_white2_and_one_colored_dice - 2] = 1
        expected_action_mask[2][12 - combined_value_of_white2_and_one_colored_dice] = 1
        expected_action_mask[3][12 - combined_value_of_white2_and_one_colored_dice] = 1

        self.force_to_use_pass_fields(expected_action_mask)

        assert_array_equal(computed_action_mask, expected_action_mask)

    def test_valid_actions_with_value_four_dices_second_part_of_round_not_tossing_player(self):
        card = GameCard("some_player_id")
        computed_action_mask = card.get_allowed_actions_mask(dices=get_dices_with_value(),
                                                             is_tossing_player=False, is_second_part_of_round=True)

        expected_action_mask = np.zeros(shape=(5, 11), dtype=int8)
        expected_action_mask[4][0:3] = 0
        expected_action_mask[4][4:] = 1
        print(expected_action_mask)

        assert_array_equal(computed_action_mask, expected_action_mask)

    def test_valid_actions_after_some_steps(self):
        card = GameCard("some_player_id")
        card._cross_value_in_line(Color.RED, 2)
        card._cross_value_in_line(Color.YELLOW, 12)
        card._cross_value_in_line(Color.GREEN, 7)

        expected_action_map = np.zeros(shape=GameCard.ACTION_MASK_SHAPE, dtype=int8)
        expected_action_map[Color.RED.value][6 - 2] = 1
        expected_action_map[Color.GREEN.value][12 - 6] = 1
        expected_action_map[Color.BLUE.value][12 - 6] = 1
        expected_action_map[4][0:4] = 0
        expected_action_map[4][4:] = 1


        computed_action_mask = card.get_allowed_actions_mask(dices=get_dices_with_value(3),
                                                             is_tossing_player=False, is_second_part_of_round=False)

        assert_array_equal(computed_action_mask, expected_action_map)

    def test_mask_for_dices(self):
        dice_value = 3
        mask = GameCard._get_mask_based_on_dices(dices=get_dices_with_value(dice_value),
                                                 is_tossing_player=False,
                                                 is_second_part_of_round=False)

        expected_action_map = np.zeros(shape=GameCard.ACTION_MASK_SHAPE, dtype=int8)
        value_of_white_dices = dice_value * 2
        expected_action_map[Color.RED.value][value_of_white_dices - 2] = 1
        expected_action_map[Color.YELLOW.value][value_of_white_dices - 2] = 1
        expected_action_map[Color.GREEN.value][12 - value_of_white_dices] = 1
        expected_action_map[Color.BLUE.value][12 - value_of_white_dices] = 1

        assert_array_equal(mask, expected_action_map)

    def test_mask_for_passes(self):
        card = GameCard("some_player_id")
        card.cross_value_with_flattened_action(44)  # add one pass
        card.crossed_something_in_current_round = False
        computed_action_mask = card.get_allowed_actions_mask(dices=get_dices_with_value(value=4),
                                                             is_tossing_player=True, is_second_part_of_round=True)

        expected_action_mask = np.zeros(shape=GameCard.ACTION_MASK_SHAPE, dtype=int8)
        combined_value_of_white_and_one_colored_dice = 8
        expected_action_mask[0][combined_value_of_white_and_one_colored_dice - 2] = 1
        expected_action_mask[1][combined_value_of_white_and_one_colored_dice - 2] = 1
        expected_action_mask[2][12 - combined_value_of_white_and_one_colored_dice] = 1
        expected_action_mask[3][12 - combined_value_of_white_and_one_colored_dice] = 1

        self.force_to_use_pass_fields(expected_action_mask)
        expected_action_mask[4][0] = 0  # set the pass from above, which can't be crossed again

        assert_array_equal(computed_action_mask, expected_action_mask)

    @staticmethod
    def force_to_use_pass_fields(expected_action_mask):
        expected_action_mask[4][0:4] = 1
        expected_action_mask[4][4:10] = 0

    def test_get_points_for_passed_field(self):
        card = GameCard("some_player_id")
        card.cross_value_with_flattened_action(44)
        self.assertEqual(-5, card.get_points())

    def test_get_points_for_blank_field(self):
        card = GameCard("some_player_id")
        card.cross_value_with_flattened_action(54)
        card.cross_value_with_flattened_action(53)
        self.assertEqual(0, card.get_points())

    def test_get_points_for_valid_field(self):
        card = GameCard("some_player_id")
        card.cross_value_with_flattened_action(0)
        card.cross_value_with_flattened_action(1)
        self.assertEqual(3, card.get_points())

    def test_get_points_for_closing_row(self):
        card = GameCard("some_player_id")
        red_row = card._rows[0]
        red_row[0:5] = 1
        card.cross_value_with_flattened_action(10)
        print(card.get_state())
        self.assertEqual(28, card.get_points())

    def test_calculate_points_for_row(self):
        points = GameCard._calculate_points_for_row(3)
        self.assertEqual(points, 6)

    def test_pass_count(self):
        card = GameCard("some_player_id")
        card._rows[4][:4] = 1

        self.assertEqual(4, card.get_pass_count())


if __name__ == '__main__':
    unittest.main()
