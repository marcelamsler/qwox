import unittest

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

        board.game_cards[players[0]]._rows[0][10] = 1
        board.game_cards[players[0]]._rows[1][10] = 1
        self.assertEqual(board.game_is_finished(), True)

    def test_game_finished_with_rows_closed_for_different_players(self):
        players = ["player1", "player2"]
        board = Board(player_ids=players)

        board.game_cards[players[0]]._rows[0][10] = 1
        board.game_cards[players[1]]._rows[1][10] = 1
        self.assertEqual(board.game_is_finished(), True)



if __name__ == '__main__':
    unittest.main()
