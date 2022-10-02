import numpy as np
from numpy import int8
import numpy.typing as npt

from game_models.color import Color


class GameCard:
    def __init__(self, player_id: str):
        self._rows: npt.NDArray = np.zeros(shape=(4, 11), dtype=int8)
        self.passes: npt.NDArray = np.zeros(shape=(1, 4), dtype=int8)
        self._player_id: str = player_id
        self.points: int = 0

    def get_allowed_actions_mask(self):
        """
        :return: np.array with shape (4,11) and 1 everywhere an action is allowed and 0 where its not allowed
        """
        mask = np.zeros(shape=(4, 11), dtype=int8)
        for mask_index, mask_row in enumerate(mask):
            row = self._rows[mask_index]
            nonzero_indexes = np.nonzero(row)
            if len(nonzero_indexes[0]):
                last_crossed_index = nonzero_indexes[0][-1]
                mask[mask_index][last_crossed_index:] = 1
                # needed because first index is inclusive
                mask[mask_index][last_crossed_index] = 0
            else:
                mask[mask_index] = 1
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
        return self._rows[row_index][-1] == 1

    def more_than_two_rows_closed(self) -> bool:
        closed_rows = 0
        for index, row in enumerate(self._rows):
            self.is_row_closed(index)

        return closed_rows > 2

    def get_state(self):
        return self._rows

    @staticmethod
    def is_reversed_line(color: Color) -> bool:
        return color == Color.GREEN or color == Color.BLUE

    def cross_value_with_flattened_action(self, action):
        index1, index2 = np.unravel_index(action, shape=(4, 11))
        self._rows[index1, index2] = 1
