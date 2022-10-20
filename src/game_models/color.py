from enum import Enum


class Color(Enum):
    WHITE = 42
    RED = 0
    YELLOW = 1
    GREEN = 2
    BLUE = 3

    def __eq__(self, other):
        return self.value == other.value

    def __repr__(self):
        return self.name.replace("Color.", "")
