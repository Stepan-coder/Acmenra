from typing import List


class Param:
    def __init__(self,
                 ptype: List[type],
                 def_val,
                 def_vals: list,
                 is_locked: bool = False,
                 min_val=None,
                 max_val=None):
        self.ptype = ptype
        self.def_val = def_val
        self.def_vals = def_vals
        self.is_locked = is_locked
        self.min_val = min_val
        self.max_val = max_val
