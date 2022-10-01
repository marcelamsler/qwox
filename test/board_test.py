import unittest

from src.game_models.board import Board


class BoardTest(unittest.TestCase):

    def test_board_creation_test(self):
        players = ["player1", "player2"]
        board = Board(player_ids=players)

        self.assertEqual(len(board.game_cards), 2)
        self.assertEqual(len(board.dices), 6)


if __name__ == '__main__':
    unittest.main()
