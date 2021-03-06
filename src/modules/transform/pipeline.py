import numpy as np
import json
import os
from . import TRANSFORM_MAP
from src.modules.utilities.logging import Logger
from collections import OrderedDict


class Pipeline:
    def __init__(self, config_path: str):
        log = Logger().logger
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file at: {config_path} not found.")
        self._config_path = config_path
        # Loading config
        self._steps = {}
        with open(config_path, mode="r") as f:
            self._steps = json.load(f)
        log.debug(f"Size of Pipeline {len(self._steps)}")

        # Ordering pipeline
        self._steps = OrderedDict(sorted(self._steps.items()))

        # Transforming to objects
        self._interpreted_list = []
        for o, v in self._steps.items():
            transform = self._steps[o]
            self._interpreted_list.append({
                "obj": TRANSFORM_MAP[transform["name"]](),
                "name": f"{transform['name']}_{o}",
                "params": transform["params"]
            })

    def fit(self, X: np.ndarray):
        log = Logger().logger
        XX = X
        for transform in self._interpreted_list:
            log.debug(f"Fitting transformation {transform['name']}...")
            XX = transform["obj"].fit(XX, **transform["params"])

    def apply(self, X: np.ndarray, path: str):
        log = Logger().logger
        XX = X
        for transform in self._interpreted_list:
            log.debug(f"Reapplying transformation {transform['name']}...")
            if transform["obj"].required_coeff:
                constructed_path = os.path.join(path, f"transforms/{transform['name']}.npy")
                log.debug(f"Transformation matrix loaded from: {constructed_path}")
                transform["obj"].coeff_ = np.load(constructed_path)
            XX = transform["obj"].apply(XX, **transform["params"])
        return XX

    def save(self, path: str):
        log = Logger().logger
        # Empty the directory first
        for path_objects in os.walk(os.path.join(path, "transforms/")):
            for file in path_objects[-1]:
                rm_path = os.path.join(os.path.join(path, "transforms/"), file)
                log.debug(f"Removing {rm_path}")
                os.remove(rm_path)

        # Saving new transform objects
        for transform in self._interpreted_list:
            if transform["obj"].coeff_ is not None:
                out_path = os.path.join(path, f"transforms/{transform['name']}.npy")
                log.debug(f"Saving transformation to {out_path}")
                np.save(out_path, transform["obj"].coeff_)
        out_path = os.path.join(path, f"transforms/config.json")
        log.debug(f"Copying configuration file to {out_path}")
        with open(out_path, mode="w") as f:
            json.dump(self._steps, f, indent=4)
