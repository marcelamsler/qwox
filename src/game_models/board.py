import numpy as np
from numpy import int8
from pettingzoo.utils.env import AgentID

from src.game_models.color import Color
from src.game_models.dice import Dice
from src.game_models.game_card import GameCard


class Board:
    TOSSING_PLAYER_OBS_INDEX = 8
    PART_OF_ROUND_OBS_INDEX = 9

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

    def get_allowed_actions_mask(self, player_id: str, is_tossing_player: bool,
                                 is_second_part_of_round: int) -> np.ndarray:
        action_mask_from_card = self.game_cards[player_id].get_allowed_actions_mask(self.dices, is_tossing_player,
                                                                                    is_second_part_of_round)
        for closed_row_index in self.get_closed_row_indexes():
            action_mask_from_card[closed_row_index] = 0

        return action_mask_from_card

    def get_observation_for_agent(self, player_id: str, is_tossing_player: bool,
                                  is_second_part_of_round: bool) -> np.ndarray:
        """
        This returns the board observation for an agent. Board observation includes the game-cards of the other players
        as well as the dice values and round part. The shape is (player_count + 1, 5,12). 5,12 is on game-card state,
        whereas the first axis represent the other players cards (plus one for dice values and state-information). The last channel represents the dice values.
        Important is that each player sees its own card on index 0 and the other ones after that.
        :param is_second_part_of_round: is it second part where coloured dices come into action?
        :param is_tossing_player: is this player tossing and allowed to use coloured dices?
        :param player_id: player-id
        """

        observation = [self.game_cards[player_id].get_state()]

        other_player_cards = [item[1].get_state() for item in self.game_cards.items() if item[0] not in player_id]
        observation.append(*other_player_cards)

        additional_information = np.zeros(shape=GameCard.OBSERVATION_SHAPE, dtype=int8)

        for idx, dice in enumerate(self.dices):
            additional_information[0][idx] = dice.current_value

        additional_information[0][self.TOSSING_PLAYER_OBS_INDEX] = 1 if is_tossing_player else 0
        additional_information[0][self.PART_OF_ROUND_OBS_INDEX] = 2 if is_second_part_of_round else 1

        observation.append(additional_information)

        return np.array(observation)
