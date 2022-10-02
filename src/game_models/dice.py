import random

from src.game_models.color import Color


class Dice:
    def __init__(self, color: Color):
        self.color: Color = color
        self.current_value: int = 1
        self.roll()

    def roll(self):
        self.current_value = random.randint(1, 6)

    def __repr__(self):
        return f"{self.color}-Dice {self.current_value}"
