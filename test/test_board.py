import unittest

import numpy as np
from numpy import int8
from numpy.testing import assert_array_equal

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

    def test_game_finished_with_passes(self):
        players = ["player1", "player2"]
        board = Board(player_ids=players)

        card: GameCard = board.game_cards[players[0]]

        card.cross_value_with_flattened_action(44)
        card.cross_value_with_flattened_action(45)
        card.cross_value_with_flattened_action(46)
        card.cross_value_with_flattened_action(47)

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
        board.dices = get_dices_with_value(5)

        crossing_player_card: GameCard = board.game_cards[players[1]]
        yellow_row = crossing_player_card._rows[1]
        yellow_row[4:9] = 1

        crossing_player_card.cross_value_with_flattened_action(21)

        other_player_card: GameCard = board.game_cards[players[0]]
        other_player_action_mask = board.get_allowed_actions_mask(player_id=players[0],
                                                                  is_tossing_player=False,
                                                                  is_second_part_of_round=False)

        count_of_allowed_actions_in_yellow_row = np.count_nonzero(other_player_action_mask[Color.YELLOW.value])

        self.assertEqual(0, count_of_allowed_actions_in_yellow_row)

    def test_get_observation_for_agent(self):
        players = ["player1", "player2"]
        board = Board(player_ids=players)
        board.dices = get_dices_with_value(2)

        first_player_card: GameCard = board.game_cards[players[0]]
        first_player_card.cross_value_with_flattened_action(0)

        second_player_card: GameCard = board.game_cards[players[1]]
        second_player_card.cross_value_with_flattened_action(1)

        expected_player1_card_observation = np.zeros(shape=GameCard.OBSERVATION_SHAPE, dtype=int8)
        expected_player1_card_observation[0][0] = 1

        expected_player2_card_observation = np.zeros(shape=GameCard.OBSERVATION_SHAPE, dtype=int8)
        expected_player2_card_observation[0][1] = 1

        expected_dice_observation = np.zeros(shape=GameCard.OBSERVATION_SHAPE, dtype=int8)
        # Set dice values
        expected_dice_observation[0][0] = 2
        expected_dice_observation[0][1] = 2
        expected_dice_observation[0][2] = 2
        expected_dice_observation[0][3] = 2
        expected_dice_observation[0][4] = 2
        expected_dice_observation[0][5] = 2

        # Set part_of_round_value and tossing_player
        expected_dice_observation[0][Board.PART_OF_ROUND_OBS_INDEX] = 2
        expected_dice_observation[0][Board.TOSSING_PLAYER_OBS_INDEX] = 1

        expected_observation_from_player1_perspective = [expected_player1_card_observation,
                                                         expected_player2_card_observation,
                                                         expected_dice_observation]

        observation_player1_perspective = board.get_observation_for_agent(players[0], is_second_part_of_round=True,
                                                                          is_tossing_player=True)

        assert_array_equal(expected_observation_from_player1_perspective[0], observation_player1_perspective[0])
        assert_array_equal(expected_observation_from_player1_perspective[1], observation_player1_perspective[1])
        assert_array_equal(expected_observation_from_player1_perspective[2], observation_player1_perspective[2])

    if __name__ == '__main__':
        unittest.main()
