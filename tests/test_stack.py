from sos_scorer.skull import Skull
from sos_scorer.scorer import Scorer

stack = {
    # Bottom Row
    0 : {
        0 : Skull.PEASANT,  # First card
        1 : Skull.PRIEST,
        2 : Skull.CRIMINAL,
        3 : Skull.PRIEST
    },
    
    0.5 : {
        0 : Skull.PRIEST,
        1 : Skull.ROMANTIC, 
        2 : Skull.ROMANTIC,
        3 : Skull.PEASANT
    },

    1 : {
        0.5 : Skull.CRIMINAL,
        1.5 : Skull.ROYAL,
        2.5 : Skull.ROMANTIC
    },

    1.5 : {
        0.5 : Skull.CRIMINAL,
        1.5 : Skull.PRIEST,
        2.5 : Skull.PEASANT
    },

    2 : {
        1 : Skull.PRIEST,
        2 : Skull.CRIMINAL
    }, 

    2.5 : {
        1 : Skull.PRIEST,
        2 : Skull.ROYAL
    }
}

print(stack)

scr = Scorer(stack)

scr.calc_score()
scr.print_scores()