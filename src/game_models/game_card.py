import itertools
from typing import Dict

import numpy as np
from numpy import int8
import numpy.typing as npt
import numpy.ma as ma
from game_models.color import Color
from game_models.dice import Dice


class GameCard:
    ACTION_MASK_SHAPE = (5, 11)
    OBSERVATION_SHAPE = (5, 11)

    def __init__(self, player_id: str):
        self._rows: npt.NDArray = np.zeros(shape=self.OBSERVATION_SHAPE, dtype=int8)
        self.passes: npt.NDArray = np.zeros(shape=(1, 4), dtype=int8)
        self._player_id: str = player_id

    def get_points(self):
        total_points = 0
        for row in self._rows:
            checked_count = np.count_nonzero(row)
            total_points += self.calculate_points_for_row(checked_count)

        total_points += np.count_nonzero(self.passes) * -5

        return total_points

    @staticmethod
    def calculate_points_for_row(checked_count: int) -> int:
        """
        The game gives points based on checked numbers in a row

        e.g.
        1 -> 1 Point
        2 -> 3 Points
        3 -> 6 Points
        ...
        12 -> 78 Points

        This can be calculated by doing 1+2+3, if the user has 3 numbers crossed
        :param checked_count: int
        :return: points : int
        """
        return sum(range(1, checked_count + 1))

    def get_allowed_actions_mask(self, dices: list[Dice], is_tossing_player: bool, is_second_part_of_round: int):
        """
        First 44 values are for values on the board, the 44th - 48th are for pass fields and 49th - 55th should do nothing
        :return: np.array with shape (4,11) and 1 everywhere an action is allowed and 0 where its not allowed

        """
        mask_based_on_crossed_numbers = self.get_mask_based_on_crossed_numbers()
        mask_based_on_dices = self.get_mask_based_on_dices(dices, is_tossing_player, is_second_part_of_round)

        combined_mask = mask_based_on_crossed_numbers & mask_based_on_dices

        self.add_pass_numbers_and_none_actions(combined_mask)

        return combined_mask

    def add_pass_numbers_and_none_actions(self, combined_mask):
        row_index = 4
        row = self._rows[row_index]
        for pass_field in range(0, 3):
            if row[row_index] == 0:
                combined_mask[row_index][pass_field] = 1

        combined_mask[row_index][3:] = 1

    def get_mask_based_on_crossed_numbers(self):
        mask_based_on_crossed_numbers = np.zeros(shape=GameCard.ACTION_MASK_SHAPE, dtype=int8)
        for mask_index, mask_row in enumerate(mask_based_on_crossed_numbers):
            if mask_index <= 3:
                row = self._rows[mask_index]
                nonzero_indexes = np.nonzero(row)
                if len(nonzero_indexes[0]):
                    last_crossed_index = nonzero_indexes[0][-1]
                    mask_based_on_crossed_numbers[mask_index][last_crossed_index:] = 1
                    # needed because first index is inclusive
                    mask_based_on_crossed_numbers[mask_index][last_crossed_index] = 0
                else:
                    mask_based_on_crossed_numbers[mask_index] = 1
        return mask_based_on_crossed_numbers

    @staticmethod
    def get_mask_based_on_dices(dices, is_tossing_player, is_second_part_of_round):
        mask = np.zeros(shape=GameCard.ACTION_MASK_SHAPE, dtype=int8)

        if is_second_part_of_round:
            if not is_tossing_player:
                return mask

            for color in [Color.RED, Color.BLUE, Color.YELLOW, Color.GREEN]:
                sum1, sum2 = GameCard.get_sums_for_color(dices, color)
                if color.value < 2:
                    mask[color.value][sum1 - 2] = 1
                    mask[color.value][sum2 - 2] = 1
                else:
                    mask[color.value][12 - sum1] = 1
                    mask[color.value][12 - sum2] = 1
        else:
            value = GameCard.get_white_dices_sum(dices)
            mask[0][value - 2] = 1
            mask[1][value - 2] = 1
            mask[2][12 - value] = 1
            mask[3][12 - value] = 1

        return mask

    def cross_value_in_line(self, line_color: Color, value: int):
        row: npt.NDArray = self._rows[line_color.value]
        if GameCard.is_reversed_line(line_color):
            row[12 - value] = 1
        else:
            row[value - 2] = 1

    def cross_value_with_action(self, action_array):
        assert np.count_nonzero(action_array) <= 1
        self._rows = self._rows + action_array

    def is_row_closed(self, row_index) -> bool:
        # TODO row can only be closed if 5 other fields are checked in that row
        return self._rows[row_index][-1] == 1

    def get_closed_row_indexes(self):
        closed_row_indexes: list[int] = []
        for index, row in enumerate(self._rows):
            if self.is_row_closed(index):
                closed_row_indexes.append(index)
        return closed_row_indexes

    def get_state(self):
        return self._rows

    @staticmethod
    def is_reversed_line(color: Color) -> bool:
        return color == Color.GREEN or color == Color.BLUE

    def cross_value_with_flattened_action(self, action):
        index1, index2 = np.array(np.unravel_index(action, shape=(5, 11)), dtype=np.intp)
        self._rows[index1, index2] = 1

    @staticmethod
    def get_white_dices_sum(dices):
        return np.sum([dice.current_value for dice in dices if dice.color == Color.WHITE])

    @staticmethod
    def get_sums_for_color(dices, color: Color) -> tuple[int, int]:
        white_dice_values = [dice.current_value for dice in dices if dice.color == Color.WHITE]
        white_dice_value1 = white_dice_values[0]
        white_dice_value2 = white_dice_values[1]
        colored_dice_value = [dice.current_value for dice in dices if dice.color == color][0]
        return white_dice_value1 + colored_dice_value, white_dice_value2 + colored_dice_value
