import json


class KDE:
    def __init__(self):
        self.kernel = None
        self.bandwidth = None

    def to_json(self) -> str:
        return json.dumps(
            {
                "kernel": self.kernel,
                "bandwidth": self.bandwidth,
            })

    def from_dict(self, params):
        self.kernel = params["kernel"]
        self.bandwidth = params["bandwidth"]

    def __str__(self):
        return self.to_json()

    def __repr__(self):
        return json.loads(self.to_json())
