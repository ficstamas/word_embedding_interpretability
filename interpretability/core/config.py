from interpretability.utils.metaclasses import Singleton
from typing import Literal
import os
import json
from .config_modules import *

__all__ = ["Config"]


class Config(metaclass=Singleton):
    def __init__(self):
        self.embedding = Embedding()
        self.semantic_categories = SemanticCategories()
        self.distance = Distance()
        self.kde = KDE()
        self.project = Project()

    def set_embedding(self, path: str, dense: bool, lines_to_read: int):
        self.embedding.path = path
        self.embedding.dense = dense
        self.embedding.lines_to_read = lines_to_read

    def set_semantic_categories(self, path: str, drop_method: Literal['random', 'category_center'],
                                drop_rate: float, seed: int):
        self.semantic_categories.path = path
        self.semantic_categories.drop_method = drop_method
        self.semantic_categories.drop_rate = drop_rate
        self.semantic_categories.seed = seed

    def set_project_path(self, path: str, name: str):
        self.project.workspace = os.path.join(os.getcwd(), path)
        self.project.name = name

    def set_kde(self, kernel: str, bandwidth: float):
        self.kde.kernel = kernel
        self.kde.bandwidth = bandwidth

    def set_distance(self, dist: Literal['bhattacharyya', 'hellinger', 'bhattacharyya_normal', 'hellinger_normal']):
        self.distance.name = dist

    def to_json(self):
        return json.dumps(
            {
                "embedding": self.embedding.__repr__(),
                "project": self.project.__repr__(),
                "kde": self.kde.__repr__(),
                "distance": self.distance.__repr__(),
                "semantic_categories": self.semantic_categories.__repr__()
            })

    def __str__(self):
        return self.to_json()

    def __repr__(self):
        return json.loads(self.__str__())
