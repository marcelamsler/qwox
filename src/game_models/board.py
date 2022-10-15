from pettingzoo.utils.env import AgentID

from src.game_models.color import Color
from src.game_models.dice import Dice
from src.game_models.game_card import GameCard


class Board:
    def __init__(self, player_ids: [str]):
        self.game_cards: dict[AgentID, GameCard] = {player: GameCard(player) for player in player_ids}
        self.dices = [Dice(color) for color in
                      [Color.RED, Color.YELLOW, Color.GREEN, Color.BLUE, Color.WHITE, Color.WHITE]]

    def roll_dices(self):
        for dice in self.dices:
            dice.roll()

    def game_is_finished(self):
        for card in self.game_cards.values():
            if card.get_pass_count() == 4:
                return True

        closed_rows_without_duplicates = self.get_closed_row_indexes()
        if len(closed_rows_without_duplicates) >= 2:
            return True

        return False

    def get_closed_row_indexes(self) -> list[int]:
        closed_rows: list[int] = []
        for card in self.game_cards.values():
            closed_rows.extend(card.get_closed_row_indexes())
        closed_rows_without_duplicates = list(dict.fromkeys(closed_rows))
        return closed_rows_without_duplicates

    def get_allowed_actions_mask(self, player_id: str, dices: list[Dice], is_tossing_player: bool,
                                 is_second_part_of_round: int):
        action_mask_from_card = self.game_cards[player_id].get_allowed_actions_mask(dices, is_tossing_player,
                                                                                    is_second_part_of_round)
        for closed_row_index in self.get_closed_row_indexes():
            action_mask_from_card[closed_row_index] = 0

        return action_mask_from_card
