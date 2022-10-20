import random

from src.game_models.color import Color


class Dice:
    def __init__(self, color: Color, current_value: int = random.randint(1, 6)):
        self.color: Color = color
        self.current_value: int = current_value

    def roll(self):
        self.current_value = random.randint(1, 6)

    def __repr__(self):
        return f"{self.color}-Dice {self.current_value}"

    def __eq__(self, other):
        return self.color.value == other.color.value and self.current_value == other.current_value