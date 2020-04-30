import json


class Embedding:
    def __init__(self):
        self.path = None
        self.dense = None
        self.lines_to_read = None

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
