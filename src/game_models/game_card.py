import numpy as np
import numpy.typing as npt
from numpy import int8

from game_models.color import Color
from game_models.dice import Dice


class GameCard:
    ACTION_MASK_SHAPE = (5, 11)
    OBSERVATION_SHAPE = (5, 12)
    OBSERVATION_SHAPE_ROWS = 5
    OBSERVATION_SHAPE_COLUMNS = 12

    def __init__(self, player_id: str):
        self._rows: npt.NDArray = np.zeros(shape=self.OBSERVATION_SHAPE, dtype=int8)
        self.crossed_something_in_current_round = False
        self._player_id: str = player_id

    def get_points(self):
        total_points = 0
        for row in self._rows[:4]:
            checked_count = np.count_nonzero(row)
            total_points += self._calculate_points_for_row(checked_count)

        total_points += self.get_pass_count() * -5

        return total_points

    def get_pass_count(self):
        return np.count_nonzero(self._rows[4][0:4])

    @staticmethod
    def _calculate_points_for_row(checked_count: int) -> int:
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
        First 44 values are for values on the board, the 44th - 48th are for pass fields and 49th - 55th are used
        for doing nothing. Doing nothing is not allowed when player hasn't taken any action but has tossed in this round

        :return: np.array with shape (4,11) and 1 everywhere an action is allowed and 0 where its not allowed
        """
        mask_based_on_crossed_numbers = self._get_mask_based_on_crossed_numbers()
        mask_based_on_dices = self._get_mask_based_on_dices(dices, is_tossing_player, is_second_part_of_round)

        combined_mask = mask_based_on_crossed_numbers & mask_based_on_dices

        final_mask = self._add_pass_numbers_and_none_actions(combined_mask, is_second_part_of_round, is_tossing_player)

        return final_mask

    def _add_pass_numbers_and_none_actions(self, combined_mask, is_second_part_of_round, is_tossing_player):
        row_index = 4
        row = self._rows[row_index]
        for pass_field in range(0, 4):
            if row[pass_field] == 0:
                combined_mask[row_index][pass_field] = 1

        allowed_to_skip_without_passing = not is_tossing_player or not is_second_part_of_round or self.crossed_something_in_current_round

        if allowed_to_skip_without_passing:
            combined_mask[row_index][3:] = 1
            # We don't want to give the option that the player crosses the pass as not learning players can't learn this
            combined_mask[row_index][0:4] = 0

        return combined_mask

    def _get_mask_based_on_crossed_numbers(self):
        mask_based_on_crossed_numbers = np.zeros(shape=GameCard.ACTION_MASK_SHAPE, dtype=int8)
        for mask_index, mask_row in enumerate(mask_based_on_crossed_numbers):
            if mask_index <= 3:
                row = self._rows[mask_index]
                nonzero_indexes = np.nonzero(row)
                if len(nonzero_indexes[0]):
                    last_crossed_index = nonzero_indexes[0][-1]
                    mask_based_on_crossed_numbers[mask_index][last_crossed_index + 1:] = 1
                else:
                    mask_based_on_crossed_numbers[mask_index] = 1
        return mask_based_on_crossed_numbers

    @staticmethod
    def _get_mask_based_on_dices(dices, is_tossing_player, is_second_part_of_round):
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

    def _cross_value_in_line(self, line_color: Color, value: int):
        row: npt.NDArray = self._rows[line_color.value]
        if GameCard.is_reversed_line(line_color):
            row[12 - value] = 1
        else:
            row[value - 2] = 1

        self.crossed_something_in_current_round = True

    def cross_value_with_flattened_action(self, action):
        """
        This allows to set the value based on a flat index. This is based on the action_space
        and not observation space, as the agent can only set on action space. (observation space has one more column
        to set a row as closed)
         Indexes:
            [[ 0  1  2  3  4  5  6  7  8  9 10]
            [11 12 13 14 15 16 17 18 19 20 21]
            [22 23 24 25 26 27 28 29 30 31 32]
            [33 34 35 36 37 38 39 40 41 42 43]
            [44 45 46 47 48 49 50 51 52 53 54]]
        :param action:
        """
        if action <= 43:
            self.crossed_something_in_current_round = True

        if action <= 47:
            row_index, column_index = np.array(np.unravel_index(action, shape=self.ACTION_MASK_SHAPE), dtype=np.intp)
            self._rows[row_index, column_index] = 1
            self._close_row_if_possible(row_index, column_index)

    def _close_row_if_possible(self, row_index, column_index):
        last_crossable_index_in_row = 10
        if column_index == last_crossable_index_in_row and np.count_nonzero(self._rows[row_index]) >= 5:
            self._rows[row_index, column_index + 1] = 1

    def _is_row_closed(self, row_index) -> bool:
        return self._rows[row_index][-1] == 1

    def get_closed_row_indexes(self):
        closed_row_indexes: list[int] = []
        for index, row in enumerate(self._rows):
            if self._is_row_closed(index):
                closed_row_indexes.append(index)
        return closed_row_indexes

    def get_state(self):
        return self._rows

    @staticmethod
    def is_reversed_line(color: Color) -> bool:
        return color == Color.GREEN or color == Color.BLUE

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
