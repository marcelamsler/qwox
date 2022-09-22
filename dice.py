from color import Color
import random


class Dice:
    def __init__(self, color: Color):
        self.color: Color = color
        self.current_value: int = 1

    def roll(self):
        self.current_value = random.randint(1, 6)
