from skull import *
from typing import Dict
from rich.console import Console
from rich.table import Table

class Scorer():
    def __init__(self, stack):
        self._stack = stack
        self._score = None

        # Internal score info
        self._sub_scores = {
            'criminals' : 0,
            'romantics' : 0,
            'priests' : 0,
            'royals' : 0,
            'peasants' : 0
        }

    def calc_score(self):
        
        self._score = 0   # total score
        self._cum_pr = 0

        lvl_data = self._create_lvl_data()
        lvl_data = dict(sorted(lvl_data.items()))

        for i, level in self._stack.items():
            for j, skull in level.items():

                if skull == Skull.PEASANT:
                    lvl_data[i]['peasants'] += 1 # Note a peasant at this level
                elif skull == Skull.PRIEST:
                    lvl_data[i]['priest'] = True # Note there is a priest in this level
                elif skull == Skull.ROYAL:
                    lvl_data[i]['royals'] += 1
                elif skull == Skull.CRIMINAL:
                    if self._adj_to_priest(i, j):
                        self._sub_scores['criminals'] += 2
                elif skull == Skull.ROMANTIC:
                    self._sub_scores['romantics'] += self._score_romantics(i, j)

        # Score the total number of peasants


        cum_royal = 0

        for lvl, data in lvl_data.items():
            self._sub_scores['royals'] += data['royals'] * cum_royal    # number of royals multiplied by the total peasants and royals below
            self._sub_scores['peasants'] += data['peasants']            # increase total number of peasants
            cum_royal += data['royals'] + data['peasants']              # increase total number of peasants and royals seen up to this level
            self._sub_scores['priests'] += 2 * data['priest']

        self._score = sum(self._sub_scores.values())
        return self._score

    def _score_romantics(self, i : float, j : float, score = True):
        romantics = 1
        self._stack[i][j] = Skull.ROMANCED

        dirs = ['left', 'right', 'up', 'down']

        for dir in dirs:
            adj_skulls = self._get_adj(i, j, dir)
            
            for skull_info in adj_skulls:
                if skull_info['skull'] == Skull.ROMANTIC:
                    _i = skull_info['idx'][0]
                    _j = skull_info['idx'][1]
                    romantics += self._score_romantics(_i, _j, score=False)
        
        if score:
            return int(romantics / 2) * 6
        else:
            return romantics
    
    def _adj_to_priest(self, i : float, j : float):
        dirs = ['left', 'right', 'up', 'down']

        for dir in dirs:
            adj_skulls = self._get_adj(i, j, dir)
            
            for skull_info in adj_skulls:
                if skull_info['skull'] == Skull.PRIEST:
                    return True

        return False

    def _get_adj(self, i : float, j : float, dir : str):

        if dir == 'left':
            adj_idxs = [(i, j-1)]
        elif dir == 'right':
            adj_idxs = [(i, j+1)]
        elif dir == 'down' or dir == 'up':
            adj_i = i - 0.5 if dir == 'down' else i + 0.5
            adj_idxs = [(adj_i, j), (adj_i, j + 0.5), (adj_i, j - 0.5)]
        else:
            raise ValueError("Direction must be: left, right, up, or down")

        adj_skulls = []
        
        for (_i, _j) in adj_idxs:

            try:
                skull = self._stack[_i][_j]
                idx = (_i, _j)

                adj_skulls.append({
                    'idx' : idx,
                    'skull' : skull
                })

            except KeyError:
                pass
        
        return adj_skulls

    
    def _create_lvl_data(self):
        return {lvl : Scorer._pr_dict() for lvl in self._stack.keys()}
    
    @staticmethod
    def _pr_dict():
        pr_dict = {
            'royals' : 0,
            'peasants' : 0,
            'priest' : False
        }

        return pr_dict


    def print_scores(self):
        """Print scores in a nice little command line table.
        """

        skull_style = {
            'royals' : "purple",
            "peasants" : "yellow",
            "priests" : "blue",
            "romantics" : "red",
            "criminals" : "white"
        }

        table = Table(title="Skulls of Sedec", show_footer=True, )

        table.add_column("Skull", footer='Total')
        table.add_column("Score", footer=str(self._score))

        for skull_name, score in self._sub_scores.items():
            table.add_row(skull_name.title(), str(score), style = skull_style[skull_name])

        console = Console()
        console.print(table)