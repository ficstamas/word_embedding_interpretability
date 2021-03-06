from interpretability.distance import bhattacharyya, hellinger
from typing import Literal
import json


class Distance:
    DISTANCES = {
        'bhattacharyya': bhattacharyya.continuous_bhattacharyya_distance,
        'hellinger': hellinger.continuous_hellinger_distance,
        'bhattacharyya_normal': bhattacharyya.closed_bhattacharyya_distance,
        'hellinger_normal': hellinger.closed_hellinger_distance,
        'hellinger_exponential': hellinger.exponential_hellinger_distance,
        'bhattacharyya_exponential': bhattacharyya.exponential_bhattacharyya_distance
    }

    def __init__(self):
        self._name = None
        self._function = None
        self._weight = None

    @property
    def weight(self):
        return self._weight

    @weight.setter
    def weight(self, val):
        if val == "none" or val == "None":
            self._weight = None
        else:
            self._weight = val

    @property
    def weight_name(self):
        return "" if self._weight is None else f"{self._weight}_"

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, val: Literal['bhattacharyya', 'hellinger', 'bhattacharyya_normal', 'hellinger_normal']):
        self._name = val
        self._function = self.DISTANCES[val]

    @property
    def function(self):
        return self._function

    def to_json(self) -> str:
        return json.dumps({"name": self.name})

    def from_dict(self, params):
        self.name = params["name"]

    def __str__(self):
        return self.to_json()

    def __repr__(self):
        return json.loads(self.to_json())
