import numpy as np
from numpy import int8
import numpy.typing as npt


class GameCard:
    def __init__(self, player_id: str):
        self.rows: npt.NDArray = np.zeros(shape=(4, 11), dtype=int8)
        self.passes: npt.NDArray = np.zeros(shape=(1, 4), dtype=int8)
        self.player_id: str = player_id
        self.points: int = 0

    def is_row_closed(self, row_index) -> bool:
        return self.rows[row_index][-1] == 1

    def more_than_two_rows_closed(self) -> bool:
        closed_rows = 0
        for index, row in enumerate(self.rows):
            self.is_row_closed(index)

        return closed_rows > 2
