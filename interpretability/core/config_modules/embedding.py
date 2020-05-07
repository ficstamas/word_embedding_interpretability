import json
# from interpretability.loader.embedding import Embedding as EmbeddingObject


class Embedding:
    def __init__(self):
        self.path = None
        self.dense = None
        self.lines_to_read = None
        self._object = None

    @property
    def embedding(self):
        """
        Returns the Embedding object ('loaders/embedding.py')
        :return:
        """
        return self._object

    @embedding.setter
    def embedding(self, val):
        self._object = val

    def to_json(self) -> str:
        return json.dumps(
            {
                "path": self.path,
                "dense": self.dense,
                "lines_to_read": self.lines_to_read,
            })

    def __str__(self):
        return self.to_json()

    def __repr__(self):
        return json.loads(self.to_json())
