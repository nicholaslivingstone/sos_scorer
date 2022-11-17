from card import Card
from typing import List

class CardSlot:
    """A slot for a card in a stack"""

    def __init__(self, card : Card, above, below, right, left):
        self._card = card
        self._above : List[Card] = above
        self._below : List[Card] = below
        self._right : Card = right
        self._left : Card = left

    @property
    def card(self):
        return self._card
    
    @property
    def above(self):
        return self._above
    
    @property
    def below(self):
        return self._below
    
    @property
    def left(self):
        return self._left

    @property
    def right(self):
        return self._right

class Stack:
    def __init__(self, card_stack):
        self._rows = []

        for card_row in card_stack[::-1]:
            row = []

            for i, card in enumerate(card_row):
                
                if card:
                   crd_slt = CardSlot(
                    card,

                   )
                    
                    

                    





        
        


