import json


class SemanticCategories:
    def __init__(self):
        self.path = None
        self.load_method = None
        self.drop_method = None
        self.drop_rate = None
        self.seed = None
        self.categories = None
        self.test_words_path = None

    def to_json(self) -> str:
        return json.dumps(
            {
                "path": self.path,
                "drop_method": self.drop_method,
                "drop_rate": self.drop_rate,
                "seed": self.seed,
                "load_method": self.load_method,
                "test_words_path": self.test_words_path,
            })

    def __str__(self):
        return self.to_json()

    def __repr__(self):
        return json.loads(self.to_json())
