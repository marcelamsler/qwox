from game_models.color import Color
from game_models.dice import Dice


def get_dices_with_value(value: int = 1) -> list[Dice]:
    return [Dice(color, value) for color in
            [Color.RED, Color.YELLOW, Color.GREEN, Color.BLUE, Color.WHITE, Color.WHITE]]
