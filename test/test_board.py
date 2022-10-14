import unittest

import numpy as np

from game_models.color import Color
from game_models.game_card import GameCard
from src.game_models.board import Board
from utils import get_dices_with_value


class BoardTest(unittest.TestCase):

    def test_board_creation_test(self):
        players = ["player1", "player2"]
        board = Board(player_ids=players)

        self.assertEqual(len(board.game_cards), 2)
        self.assertEqual(len(board.dices), 6)

    def test_game_finished_with_rows_closed(self):
        players = ["player1", "player2"]
        board = Board(player_ids=players)

        card: GameCard = board.game_cards[players[0]]
        red_row = card._rows[0]
        red_row[0:6] = 1
        yellow_row = card._rows[1]
        yellow_row[4:9] = 1

        card.cross_value_with_flattened_action(10)
        card.cross_value_with_flattened_action(21)

        self.assertEqual(board.game_is_finished(), True)

    def test_game_finished_with_rows_closed_for_different_players(self):
        players = ["player1", "player2"]
        board = Board(player_ids=players)

        player1_card: GameCard = board.game_cards[players[0]]
        player1_card._rows[0][0:6] = 1

        player2_card: GameCard = board.game_cards[players[1]]
        player2_card._rows[1][4:9] = 1

        player1_card.cross_value_with_flattened_action(10)
        player2_card.cross_value_with_flattened_action(21)

        self.assertEqual(board.game_is_finished(), True)

    def test_row_closing_for_others(self):
        players = ["player1", "player2"]
        board = Board(player_ids=players)

        crossing_player_card: GameCard = board.game_cards[players[1]]
        yellow_row = crossing_player_card._rows[1]
        yellow_row[4:9] = 1

        crossing_player_card.cross_value_with_flattened_action(21)

        other_player_card: GameCard = board.game_cards[players[0]]
        other_player_action_mask = board.get_allowed_actions_mask(player_id=players[0], dices=get_dices_with_value(5),
                                                                  is_tossing_player=False,
                                                                  is_second_part_of_round=False)

        count_of_allowed_actions_in_yellow_row = np.count_nonzero(other_player_action_mask[Color.YELLOW.value])

        self.assertEqual(0, count_of_allowed_actions_in_yellow_row)

    if __name__ == '__main__':
        unittest.main()
