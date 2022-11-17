from skull import *

class Card:
    def __init__(self, top : Skull, bottom : Skull):
        self._top = top
        self._bottom = bottom
  

    @property
    def top(self):
        return self._top

    @property
    def bottom(self):
        return self._bottom