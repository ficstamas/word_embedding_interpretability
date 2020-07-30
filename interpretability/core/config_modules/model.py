import json


class ModelParams:
    def __init__(self):
        self.mcmc_acceptance = 200
        self.mcmc_noise = 0.3

    def to_json(self) -> str:
        return json.dumps(
            {
                "mcmc_acceptance": self.mcmc_acceptance,
                "mcmc_noise": self.mcmc_noise,
            })

    def from_dict(self, params):
        self.mcmc_acceptance = params["mcmc_acceptance"]
        self.mcmc_noise = params["mcmc_noise"]

    def __str__(self):
        return self.to_json()

    def __repr__(self):
        return json.loads(self.to_json())
