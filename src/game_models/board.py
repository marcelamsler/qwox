import numpy as np
from pettingzoo.utils.env import AgentID

from src.game_models.color import Color
from src.game_models.dice import Dice
from src.game_models.game_card import GameCard


class Board:
    def __init__(self, player_ids: [str]):
        self.game_cards: {AgentID: GameCard} = {player: GameCard(player) for player in player_ids}
        self.dices = [Dice(color) for color in
                      [Color.RED, Color.YELLOW, Color.GREEN, Color.BLUE, Color.WHITE, Color.WHITE]]

    def roll_dices(self):
        for dice in self.dices:
            dice.roll()

    def game_is_finished(self):
        for card in self.game_cards.values():
            if np.count_nonzero(card.passes) == 4 or card.more_than_two_rows_closed():
                return True
        return False
