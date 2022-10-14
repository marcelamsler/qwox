import unittest

from game_models.game_card import GameCard
from src.game_models.board import Board


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

        card: GameCard = board.game_cards[players[0]]
        red_row = card._rows[0]
        red_row[0:6] = 1

        card2: GameCard = board.game_cards[players[1]]
        yellow_row = card._rows[1]
        yellow_row[4:9] = 1

        card.cross_value_with_flattened_action(10)
        card2.cross_value_with_flattened_action(21)

        self.assertEqual(board.game_is_finished(), True)


if __name__ == '__main__':
    unittest.main()
