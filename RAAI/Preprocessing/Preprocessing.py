import os
import json
import math
from typing import List

import numpy as np
import pandas as pd


class Encoder:
    def __init__(self):
        self.__unique = None
        self.__labels_count = None
        self.__labels_dict = {}
        self.__labels_dict_r = {}

    def fit(self, values: List[bool or int or float or str], ):
        if len(values) <= 0:
            raise Exception("The length of the training sample must be greater than 0!")
        for i in range(len(values)):
            if isinstance(values[i], list):
                raise Exception(f"The values must be hashable type! Error on the value '{values[i]}'! ")
        self.__unique = list(set(values))
        for i in range(len(self.__unique)):
            self.__labels_dict[self.__unique[i]] = i
            self.__labels_dict_r[i] = self.__unique[i]

    def encode(self, val):
        if isinstance(val, list):
            if len(val) <= 0:
                raise Exception("The length of the training sample must be greater than 0!")
            values = []
            for v in val:
                values.append(self.__labels_dict[v])
            return values
        else:
            return self.__labels_dict[val]

    def decode(self, val):
        if isinstance(val, list):
            if len(val) <= 0:
                raise Exception("The length of the training sample must be greater than 0!")
            values = []
            for v in val:
                values.append(self.__labels_dict_r[v])
            return values
        else:
            return self.__labels_dict_r[val]

